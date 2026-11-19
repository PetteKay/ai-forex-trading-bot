"""
Trade Journal & Observability Layer.

Logs every bot decision, trade, and outcome to a SQLite journal.
Separate from the memory store — this is the audit trail and analytics source.

Tables:
  ticks       — every bot tick (market state + decision)
  trades      — every opened/closed trade with full context
  equity_log  — equity snapshot every N ticks

Usage:
  from observability.journal import Journal
  journal = Journal("observability/trade_journal.db")
  journal.log_tick(...)
  journal.log_trade_open(...)
  journal.log_trade_close(...)
"""

import sqlite3
from sqlite_conn import connect as _sqlite_connect
import json
import os
import logging
from datetime import datetime, timezone

log = logging.getLogger("Journal")


class Journal:
    def __init__(self, db_path: str = "observability/trade_journal.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = db_path
        self._init_db()
        log.info(f"Journal ready: {db_path}")

    def log_tick(
        self,
        symbol: str,
        timeframe: str,
        price: float,
        regime: str,
        tech_bias: str,
        pm_action: str,
        pm_confidence: int,
        governor_approved: bool,
        meta_label_prob: float,
        session: str,
        fingerprint: str,
        ctx_json: dict = None,
    ) -> None:
        with _sqlite_connect(self._db) as con:
            con.execute("""
                INSERT INTO ticks (
                    timestamp, symbol, timeframe, price, regime, tech_bias,
                    pm_action, pm_confidence, governor_approved, meta_label_prob,
                    session, fingerprint, ctx_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                symbol, timeframe, price, regime, tech_bias,
                pm_action, pm_confidence, int(governor_approved), meta_label_prob,
                session, fingerprint,
                json.dumps(ctx_json) if ctx_json else None,
            ))

    def log_trade_open(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        volume: float,
        confidence: int,
        regime: str,
        fingerprint: str,
        size_multiplier: float = 1.0,
    ) -> None:
        with _sqlite_connect(self._db) as con:
            con.execute("""
                INSERT OR REPLACE INTO trades (
                    ticket, symbol, direction, entry_price, sl, tp, volume,
                    confidence, regime, fingerprint, size_multiplier,
                    entry_time, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ticket, symbol, direction, entry_price, sl, tp, volume,
                confidence, regime, fingerprint, size_multiplier,
                datetime.now(timezone.utc).isoformat(), "OPEN",
            ))

    def log_trade_close(
        self,
        ticket: int,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        outcome: str,
        hold_hours: float,
        exit_reason: str = "UNKNOWN",
    ) -> None:
        with _sqlite_connect(self._db) as con:
            con.execute("""
                UPDATE trades SET
                    exit_price=?, pnl=?, pnl_pct=?, outcome=?,
                    hold_hours=?, exit_reason=?, exit_time=?, status=?
                WHERE ticket=?
            """, (
                exit_price, pnl, pnl_pct, outcome,
                hold_hours, exit_reason,
                datetime.now(timezone.utc).isoformat(), "CLOSED",
                ticket,
            ))

    def log_equity(self, equity: float, daily_pnl: float, open_positions: int) -> None:
        with _sqlite_connect(self._db) as con:
            con.execute("""
                INSERT INTO equity_log (timestamp, equity, daily_pnl, open_positions)
                VALUES (?,?,?,?)
            """, (datetime.now(timezone.utc).isoformat(), equity, daily_pnl, open_positions))

    def get_recent_trades(self, n: int = 20) -> list[dict]:
        with _sqlite_connect(self._db) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_equity_curve(self) -> list[dict]:
        with _sqlite_connect(self._db) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT timestamp, equity FROM equity_log ORDER BY timestamp"
            ).fetchall()
        return [dict(r) for r in rows]

    def print_summary(self) -> None:
        with _sqlite_connect(self._db) as con:
            total   = con.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'").fetchone()[0]
            wins    = con.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0").fetchone()[0]
            avg_pnl = con.execute("SELECT AVG(pnl) FROM trades WHERE status='CLOSED'").fetchone()[0] or 0
            best    = con.execute("SELECT MAX(pnl_pct) FROM trades").fetchone()[0] or 0
            worst   = con.execute("SELECT MIN(pnl_pct) FROM trades").fetchone()[0] or 0

        win_rate = wins / total * 100 if total else 0
        print(f"\n{'='*50}")
        print(f"  Journal Summary: {total} closed trades")
        print(f"  Win rate: {win_rate:.1f}% | Avg P&L: {avg_pnl:.2f}")
        print(f"  Best: {best:+.2f}% | Worst: {worst:+.2f}%")
        print(f"{'='*50}\n")

    # ── Setup Blacklist ──────────────────────────────────────────────────────

    def add_blacklist_loss(self, symbol: str, regime: str) -> int:
        """Record a consecutive loss for this symbol+regime. Returns new count."""
        with _sqlite_connect(self._db) as con:
            con.execute("""
                INSERT INTO setup_blacklist (symbol, regime, consecutive_losses, last_loss_time)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(symbol, regime) DO UPDATE SET
                    consecutive_losses = consecutive_losses + 1,
                    last_loss_time     = excluded.last_loss_time
            """, (symbol.upper(), regime, datetime.now(timezone.utc).isoformat()))
            row = con.execute(
                "SELECT consecutive_losses FROM setup_blacklist WHERE symbol=? AND regime=?",
                (symbol.upper(), regime)
            ).fetchone()
        return row[0] if row else 1

    def reset_blacklist(self, symbol: str, regime: str) -> None:
        """Reset consecutive loss counter after a win."""
        with _sqlite_connect(self._db) as con:
            con.execute("""
                UPDATE setup_blacklist SET consecutive_losses=0
                WHERE symbol=? AND regime=?
            """, (symbol.upper(), regime))

    def is_blacklisted(self, symbol: str, regime: str, threshold: int = 3) -> bool:
        """Return True if this symbol+regime has >= threshold consecutive losses."""
        try:
            with _sqlite_connect(self._db) as con:
                row = con.execute(
                    "SELECT consecutive_losses FROM setup_blacklist WHERE symbol=? AND regime=?",
                    (symbol.upper(), regime)
                ).fetchone()
            return bool(row and row[0] >= threshold)
        except Exception:
            return False

    # ── Agent Accuracy Tracking ──────────────────────────────────────────────

    def log_agent_votes(
        self,
        ticket: int,
        tech_bias: str,
        tech_strength: int,
        bear_risk: int,
        pm_action: str,
        pm_confidence: int,
        regime: str = "",
    ) -> None:
        """Record agent votes at trade open. Outcome updated when trade closes."""
        with _sqlite_connect(self._db) as con:
            con.execute("""
                INSERT OR IGNORE INTO agent_votes
                    (ticket, tech_bias, tech_strength, bear_risk, pm_action, pm_confidence, regime, voted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticket, tech_bias, tech_strength, bear_risk,
                pm_action, pm_confidence, regime,
                datetime.now(timezone.utc).isoformat(),
            ))

    def update_agent_outcome(self, ticket: int, is_win: bool) -> None:
        """Mark the trade outcome so accuracy can be computed."""
        with _sqlite_connect(self._db) as con:
            con.execute("""
                UPDATE agent_votes SET outcome_is_win=? WHERE ticket=?
            """, (1 if is_win else 0, ticket))

    def get_agent_accuracy(self, lookback: int = 50) -> dict:
        """
        Compute per-agent accuracy over last N completed trades.

        Returns dict with:
          tech_accuracy:      float — % trades where tech_bias matched outcome
          bear_accuracy:      float — % trades where high bear_risk (>=6) correctly predicted a loss
          pm_accuracy:        float — correlation between pm_confidence and win_rate buckets
          n_trades:           int
        """
        try:
            with _sqlite_connect(self._db) as con:
                rows = con.execute("""
                    SELECT tech_bias, tech_strength, bear_risk, pm_action, pm_confidence, outcome_is_win
                    FROM agent_votes
                    WHERE outcome_is_win IS NOT NULL
                    ORDER BY voted_at DESC LIMIT ?
                """, (lookback,)).fetchall()
        except Exception:
            return {"tech_accuracy": 0.5, "bear_accuracy": 0.5, "pm_accuracy": 0.5, "n_trades": 0}

        if not rows:
            return {"tech_accuracy": 0.5, "bear_accuracy": 0.5, "pm_accuracy": 0.5, "n_trades": 0}

        n          = len(rows)
        tech_right = 0
        bear_right = 0
        bear_total = 0
        high_conf_wins = 0
        high_conf_total = 0

        for tech_bias, tech_str, bear_risk, pm_action, pm_conf, is_win in rows:
            # Tech right if bias == action and trade won, OR bias != action and trade lost
            bias_matches_action = tech_bias and pm_action and tech_bias[:3] == pm_action[:3]
            if (bias_matches_action and is_win) or (not bias_matches_action and not is_win):
                tech_right += 1

            # Bear right if bear_risk >= 6 and trade lost
            if bear_risk is not None and bear_risk >= 6:
                bear_total += 1
                if not is_win:
                    bear_right += 1

            # PM right if high confidence (>=65) + trade won
            if pm_conf is not None and pm_conf >= 65:
                high_conf_total += 1
                if is_win:
                    high_conf_wins += 1

        tech_acc  = round(tech_right / n, 3) if n else 0.5
        bear_acc  = round(bear_right / bear_total, 3) if bear_total else 0.5
        pm_acc    = round(high_conf_wins / high_conf_total, 3) if high_conf_total else 0.5

        return {
            "tech_accuracy": tech_acc,
            "bear_accuracy": bear_acc,
            "pm_accuracy":   pm_acc,
            "n_trades":      n,
        }

    def _init_db(self):
        with _sqlite_connect(self._db) as con:
            # WAL mode prevents "database is locked" when equity_chart.py reads
            # concurrently while the bot is writing.
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp         TEXT,
                    symbol            TEXT,
                    timeframe         TEXT,
                    price             REAL,
                    regime            TEXT,
                    tech_bias         TEXT,
                    pm_action         TEXT,
                    pm_confidence     INTEGER,
                    governor_approved INTEGER,
                    meta_label_prob   REAL,
                    session           TEXT,
                    fingerprint       TEXT,
                    ctx_json          TEXT
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    ticket            INTEGER PRIMARY KEY,
                    symbol            TEXT,
                    direction         TEXT,
                    entry_price       REAL,
                    exit_price        REAL,
                    sl                REAL,
                    tp                REAL,
                    volume            REAL,
                    confidence        INTEGER,
                    regime            TEXT,
                    fingerprint       TEXT,
                    size_multiplier   REAL DEFAULT 1.0,
                    pnl               REAL,
                    pnl_pct           REAL,
                    outcome           TEXT,
                    hold_hours        REAL,
                    exit_reason       TEXT,
                    entry_time        TEXT,
                    exit_time         TEXT,
                    status            TEXT DEFAULT 'OPEN'
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS equity_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT,
                    equity          REAL,
                    daily_pnl       REAL,
                    open_positions  INTEGER
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS setup_blacklist (
                    symbol              TEXT NOT NULL,
                    regime              TEXT NOT NULL DEFAULT '',
                    consecutive_losses  INTEGER DEFAULT 0,
                    last_loss_time      TEXT,
                    PRIMARY KEY (symbol, regime)
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS agent_votes (
                    ticket          INTEGER PRIMARY KEY,
                    tech_bias       TEXT,
                    tech_strength   INTEGER,
                    bear_risk       INTEGER,
                    pm_action       TEXT,
                    pm_confidence   INTEGER,
                    regime          TEXT,
                    voted_at        TEXT,
                    outcome_is_win  INTEGER
                )
            """)
