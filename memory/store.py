"""
Selective Trade Memory Store — v2 with embedding-based similarity.

Philosophy: quality over quantity.
- Fingerprint matching (fast, exact/partial) for session recall
- Feature-vector cosine similarity (richer, for training and deeper recall)
- Only stores memories above the usefulness threshold
- Prunes weakest memories when over capacity
"""

import sqlite3
import json
import math
import os
import logging
import numpy as np
from datetime import datetime, timezone

log = logging.getLogger("Memory")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class TradeMemoryStore:
    def __init__(self, db_path: str, usefulness_threshold: float = 0.55, max_stored: int = 1000):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db     = db_path
        self._thresh = usefulness_threshold
        self._max    = max_stored
        self._init_db()
        log.info(f"Memory store ready: {db_path} (threshold={usefulness_threshold})")

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, memory: dict) -> bool:
        """
        Save a trade memory if it clears the usefulness threshold.
        Returns True if saved.
        """
        score = float(memory.get("usefulness_score", 0))
        if score < self._thresh:
            log.info(f"Memory rejected (score={score:.2f} < {self._thresh}). "
                     f"Lesson: {memory.get('lesson','')[:60]}")
            return False

        fv = memory.get("feature_vector")
        fv_json = json.dumps(fv) if fv else None

        # Semantic embedding of lesson text (best-effort — never blocks save)
        lesson_emb_json = None
        try:
            from memory.semantic import embed_text
            emb = embed_text(memory["lesson"])
            if emb:
                lesson_emb_json = json.dumps(emb)
        except Exception as e:
            log.debug(f"Semantic embedding skipped: {e}")

        with sqlite3.connect(self._db) as con:
            con.execute("""
                INSERT INTO memories
                  (timestamp, symbol, direction, entry_price, exit_price,
                   pnl_pct, r_multiple, outcome, fingerprint, feature_vector,
                   market_context, agent_reasoning, lesson, usefulness_score,
                   pattern_name, category, times_recalled, lesson_embedding,
                   regime, fib_proximity, cot_z_score, econ_surprise,
                   entry_time, ml_sampled)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                memory["symbol"],
                memory["direction"],
                memory["entry_price"],
                memory["exit_price"],
                memory["pnl_pct"],
                memory.get("r_multiple"),
                memory["outcome"],
                memory["fingerprint"],
                fv_json,
                json.dumps(memory.get("market_context", {})),
                memory.get("agent_reasoning", ""),
                memory["lesson"],
                score,
                memory.get("pattern_name"),
                memory.get("category"),
                lesson_emb_json,
                memory.get("regime", ""),
                float(memory.get("fib_proximity", 0.0) or 0.0),
                float(memory.get("cot_z_score", 0.0) or 0.0),
                float(memory.get("econ_surprise", 0.0) or 0.0),
                memory.get("entry_time"),
                int(memory.get("ml_sampled", 0) or 0),
            ))

        log.info(
            f"Memory saved (score={score:.2f}): "
            f"{memory['direction']} {memory['symbol']} → {memory['outcome']} | "
            f"{memory['lesson'][:80]}"
        )
        self._prune_if_needed()
        return True

    def recall(self, fingerprint: str, symbol: str, n: int = 5,
               feature_vector: list[float] = None) -> list[dict]:
        """
        Retrieve the N most relevant past memories.
        Uses fingerprint matching first; if feature_vector is supplied,
        re-ranks candidates by cosine similarity.
        """
        with sqlite3.connect(self._db) as con:
            con.row_factory = sqlite3.Row

            rows = con.execute("""
                SELECT * FROM memories
                WHERE symbol = ? AND fingerprint = ?
                ORDER BY usefulness_score DESC, times_recalled ASC
                LIMIT ?
            """, (symbol, fingerprint, n)).fetchall()

            if len(rows) < n:
                already = {r["id"] for r in rows}
                partial = con.execute("""
                    SELECT * FROM memories
                    WHERE symbol = ?
                    ORDER BY usefulness_score DESC
                    LIMIT ?
                """, (symbol, n * 5)).fetchall()

                for row in partial:
                    if row["id"] in already:
                        continue
                    sim = self._fingerprint_similarity(fingerprint, row["fingerprint"])
                    if sim >= 0.625:  # 5/8 chars
                        rows = list(rows) + [row]
                        already.add(row["id"])
                    if len(rows) >= n * 2:
                        break

            ids = [r["id"] for r in rows[:n * 2]]
            if ids:
                con.execute(
                    f"UPDATE memories SET times_recalled = times_recalled + 1 "
                    f"WHERE id IN ({','.join('?'*len(ids))})",
                    ids
                )

        candidates = list(rows[:n * 2])

        # Re-rank by feature vector cosine similarity + time decay
        if candidates:
            scored = []
            now_ts = datetime.now(timezone.utc)
            for row in candidates:
                fv_stored = None
                try:
                    raw = row["feature_vector"]
                    if raw:
                        fv_stored = json.loads(raw)
                except Exception:
                    pass
                # Missing vector = no similarity (0.0), not neutral (0.5 was wrong —
                # neutral falsely boosts memories with no feature data)
                cos_sim = _cosine_similarity(feature_vector, fv_stored) if (feature_vector and fv_stored) else 0.0

                # Time-decay: memories older than 30 days are progressively down-weighted.
                # Verified finding: stale memories from different regimes actively degrade
                # performance (concept drift). Half-life = 60 days.
                age_days = 0.0
                try:
                    mem_dt   = datetime.fromisoformat(row["timestamp"])
                    age_days = max(0.0, (now_ts - mem_dt).total_seconds() / 86400)
                except Exception:
                    pass
                decay = math.exp(-age_days / 60.0)  # half-life ~60 days

                usefulness  = float(row["usefulness_score"])
                final_score = usefulness * 0.5 + cos_sim * 0.3 + decay * 0.2
                scored.append((final_score, row))
            scored.sort(key=lambda x: x[0], reverse=True)
            candidates = [r for _, r in scored]

        result = []
        for row in candidates[:n]:
            result.append({
                "direction":        row["direction"],
                "outcome":          row["outcome"],
                "pnl_pct":          row["pnl_pct"],
                "lesson":           row["lesson"],
                "pattern_name":     row["pattern_name"],
                "usefulness_score": row["usefulness_score"],
                "timestamp":        row["timestamp"],
                "category":         row["category"],
            })

        log.debug(f"Recalled {len(result)} memories for fingerprint={fingerprint}")
        return result

    def recall_balanced(self, fingerprint: str, symbol: str, n: int = 5,
                        feature_vector: list[float] = None,
                        min_losses: int = 2) -> list[dict]:
        """
        Retrieve N memories ensuring at least min_losses are from losing trades.

        Rationale (arXiv:2507.23211): including negative exemplars in LLM few-shot
        produces up to 16.3% performance improvement. Standard recall() skews to
        winners because they have higher usefulness_score and survive pruning better.
        Balanced recall forces the LLM desk to see the boundary — what NOT to do
        matters as much as what works.

        Strategy:
          1. Pull a large candidate pool (n × 4) using the normal scoring pipeline
          2. Partition by outcome into win pool and loss pool
          3. Allocate min_losses slots from loss pool (best-scored losses first)
          4. Fill remaining slots from win pool
          5. Re-sort by final_score for display (don't show them grouped by outcome)
        """
        _WIN_OUTCOMES  = {"WIN", "WIN_STRONG", "PARTIAL_WIN", "BREAKEVEN"}
        _LOSS_OUTCOMES = {"LOSS", "STOP_LOSS", "TIMEOUT_LOSS", "PARTIAL_LOSS",
                          "LOSS_MANAGED", "CLOSED_LOSS"}

        # Pull a large candidate pool
        pool = self.recall(
            fingerprint, symbol, n=n * 4,
            feature_vector=feature_vector
        )

        if len(pool) <= n:
            # Not enough memories to balance — return what we have
            return pool

        win_pool  = [m for m in pool if m.get("outcome") in _WIN_OUTCOMES]
        loss_pool = [m for m in pool if m.get("outcome") in _LOSS_OUTCOMES]
        other     = [m for m in pool if m not in win_pool and m not in loss_pool]

        actual_min_losses = min(min_losses, len(loss_pool), n - 1)
        slots_for_wins    = n - actual_min_losses

        selected = loss_pool[:actual_min_losses] + win_pool[:slots_for_wins]

        # Fill any remaining slots with other/unlabeled
        while len(selected) < n and other:
            selected.append(other.pop(0))

        # Re-sort by usefulness_score so the LLM sees best examples first
        selected.sort(key=lambda m: m.get("usefulness_score", 0), reverse=True)

        log.debug(
            f"recall_balanced: {len(selected)} memories "
            f"({actual_min_losses} losses, {len(selected)-actual_min_losses} wins/other)"
        )
        return selected[:n]

    def semantic_recall(self, query: str, symbol: str, n: int = 3) -> list[dict]:
        """
        Retrieve memories whose lesson text is semantically similar to query.
        Uses NVIDIA embedding API. Returns [] if embeddings unavailable.
        """
        try:
            from memory.semantic import embed_text
            q_emb = embed_text(query)
            if not q_emb:
                return []
        except Exception:
            return []

        with sqlite3.connect(self._db) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM memories WHERE symbol = ? AND lesson_embedding IS NOT NULL",
                (symbol,)
            ).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            try:
                stored_emb = json.loads(row["lesson_embedding"])
                sim = _cosine_similarity(q_emb, stored_emb)
                scored.append((sim, row))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for sim, row in scored[:n]:
            if sim < 0.5:
                continue
            result.append({
                "direction":        row["direction"],
                "outcome":          row["outcome"],
                "pnl_pct":          row["pnl_pct"],
                "lesson":           row["lesson"],
                "pattern_name":     row["pattern_name"],
                "usefulness_score": row["usefulness_score"],
                "timestamp":        row["timestamp"],
                "category":         row["category"],
                "semantic_similarity": round(sim, 3),
            })

        log.debug(f"Semantic recall: {len(result)} results for query='{query[:60]}'")
        return result

    def save_ml_sample(self, sample: dict) -> None:
        """
        Store one closed trade as ML training data — every closed trade, with no
        usefulness filter and no pruning. Idempotent per ticket.
        """
        with sqlite3.connect(self._db) as con:
            con.execute("""
                INSERT OR IGNORE INTO ml_samples
                  (ticket, symbol, direction, entry_time, exit_time, feature_vector,
                   regime, fib_proximity, cot_z_score, econ_surprise,
                   r_multiple, pnl_pct, outcome)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sample.get("ticket"),
                sample["symbol"],
                sample["direction"],
                sample["entry_time"],
                sample.get("exit_time"),
                json.dumps(sample["feature_vector"]),
                sample.get("regime", ""),
                float(sample.get("fib_proximity", 0.0) or 0.0),
                float(sample.get("cot_z_score", 0.0) or 0.0),
                float(sample.get("econ_surprise", 0.0) or 0.0),
                sample.get("r_multiple"),
                sample.get("pnl_pct"),
                sample.get("outcome"),
            ))

    def get_all_for_training(self) -> list[dict]:
        """
        Return training rows for the meta-label model:
          • every live closed trade from ml_samples (unfiltered), plus
          • memories not already covered there — historical-trainer seeds and
            lessons saved before ml_samples existed (ml_sampled = 0).
        Live trades whose lesson was also saved (ml_sampled = 1) appear once.
        """
        with sqlite3.connect(self._db) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM memories WHERE COALESCE(ml_sampled, 0) = 0 "
                "ORDER BY timestamp DESC"
            ).fetchall()
            ml_rows = con.execute(
                "SELECT * FROM ml_samples ORDER BY entry_time DESC"
            ).fetchall()
        samples = [
            {
                "symbol":         r["symbol"],
                "direction":      r["direction"],
                "outcome":        r["outcome"],
                "pnl_pct":        r["pnl_pct"],
                "r_multiple":     r["r_multiple"],
                "feature_vector": json.loads(r["feature_vector"]) if r["feature_vector"] else None,
                "timestamp":      r["entry_time"],
                "entry_time":     r["entry_time"],
                "exit_time":      r["exit_time"],
                "regime":         r["regime"] or "",
                "fib_proximity":  r["fib_proximity"] or 0.0,
                "cot_z_score":    r["cot_z_score"] or 0.0,
                "econ_surprise":  r["econ_surprise"] or 0.0,
                "source":         "ml_samples",
            }
            for r in ml_rows
        ]
        return samples + [
            {
                "symbol":           r["symbol"],       # needed for per-symbol uniqueness weighting
                "direction":        r["direction"],
                "outcome":          r["outcome"],
                "pnl_pct":          r["pnl_pct"],
                "r_multiple":       r["r_multiple"],   # preferred label for ML training
                "fingerprint":      r["fingerprint"],
                "feature_vector":   json.loads(r["feature_vector"]) if r["feature_vector"] else None,
                "usefulness_score": r["usefulness_score"],
                "timestamp":        r["timestamp"],
                "lesson":           r["lesson"],
                "category":         r["category"],
                "regime":           r["regime"] or "",
                "fib_proximity":    r["fib_proximity"] or 0.0,
                "cot_z_score":      r["cot_z_score"] or 0.0,
                "econ_surprise":    r["econ_surprise"] or 0.0,
                "entry_time":       r["entry_time"],
                "source":           "memories",
            }
            for r in rows
        ]

    def stats(self) -> dict:
        with sqlite3.connect(self._db) as con:
            total  = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            avg_u  = con.execute("SELECT AVG(usefulness_score) FROM memories").fetchone()[0] or 0
            outcomes = dict(con.execute(
                "SELECT outcome, COUNT(*) FROM memories GROUP BY outcome"
            ).fetchall())
        return {"total_memories": total, "avg_usefulness": round(avg_u, 2), "by_outcome": outcomes}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self._db) as con:
            # WAL mode: allows concurrent readers while writer is active;
            # prevents "database is locked" under the 60s bot tick loop.
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp        TEXT NOT NULL,
                    symbol           TEXT NOT NULL,
                    direction        TEXT NOT NULL,
                    entry_price      REAL,
                    exit_price       REAL,
                    pnl_pct          REAL,
                    r_multiple       REAL,
                    outcome          TEXT,
                    fingerprint      TEXT,
                    feature_vector   TEXT,
                    market_context   TEXT,
                    agent_reasoning  TEXT,
                    lesson           TEXT,
                    usefulness_score REAL,
                    pattern_name     TEXT,
                    category         TEXT,
                    times_recalled   INTEGER DEFAULT 0
                )
            """)
            # Migration: add columns to existing DBs created before this schema version
            for col_def in [
                ("ALTER TABLE memories ADD COLUMN feature_vector TEXT", None),
                ("ALTER TABLE memories ADD COLUMN r_multiple REAL",     None),
                ("ALTER TABLE memories ADD COLUMN lesson_embedding TEXT", None),
                # ML context columns — bot.py has always sent these on save();
                # without the columns they were silently dropped, so the meta-label
                # model trained with regime/fib/COT/econ permanently zeroed.
                ("ALTER TABLE memories ADD COLUMN regime TEXT",          None),
                ("ALTER TABLE memories ADD COLUMN fib_proximity REAL",   None),
                ("ALTER TABLE memories ADD COLUMN cot_z_score REAL",     None),
                ("ALTER TABLE memories ADD COLUMN econ_surprise REAL",   None),
                # Trade entry time (time features must use entry, not save time)
                ("ALTER TABLE memories ADD COLUMN entry_time TEXT",      None),
                # 1 = this trade's training row lives in ml_samples
                ("ALTER TABLE memories ADD COLUMN ml_sampled INTEGER DEFAULT 0", None),
            ]:
                try:
                    con.execute(col_def[0])
                except Exception:
                    pass
            # Every closed live trade, unfiltered — meta-label training data.
            con.execute("""
                CREATE TABLE IF NOT EXISTS ml_samples (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket         INTEGER UNIQUE,
                    symbol         TEXT NOT NULL,
                    direction      TEXT NOT NULL,
                    entry_time     TEXT NOT NULL,
                    exit_time      TEXT,
                    feature_vector TEXT NOT NULL,
                    regime         TEXT,
                    fib_proximity  REAL,
                    cot_z_score    REAL,
                    econ_surprise  REAL,
                    r_multiple     REAL,
                    pnl_pct        REAL,
                    outcome        TEXT
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_ml_entry ON ml_samples(entry_time)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_symbol      ON memories(symbol)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint ON memories(fingerprint)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_timestamp   ON memories(timestamp)")

    def _prune_if_needed(self):
        with sqlite3.connect(self._db) as con:
            total = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if total > self._max * 1.2:
                excess = total - self._max
                con.execute(
                    "DELETE FROM memories WHERE id IN ("
                    "  SELECT id FROM memories"
                    "  ORDER BY (usefulness_score + times_recalled * 0.1) ASC"
                    "  LIMIT ?"
                    ")",
                    (excess,),
                )
                log.info(f"Pruned memory to {self._max} entries.")

    @staticmethod
    def _fingerprint_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        length  = min(len(a), len(b))
        matches = sum(1 for i in range(length) if a[i] == b[i])
        return matches / length
