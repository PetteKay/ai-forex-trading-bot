"""
Trade Pattern Miner — statistical edge analysis from journal history.

Mines closed trades to find which symbol × regime × session combinations
have the highest win rate, then produces a runtime confidence boost/penalty
for setups that are statistically strong or weak.

Results are cached for CACHE_TICKS ticks so we don't hit SQLite every tick.
"""

import sqlite3
from sqlite_conn import connect as _sqlite_connect
import logging
from collections import defaultdict
from datetime import datetime

log = logging.getLogger("PatternMiner")

_patterns_cache: dict   = {}
_cache_trade_count: int = 0
CACHE_TICKS = 10        # re-mine every 10 new closed trades


def _session_from_time(entry_time: str) -> str:
    try:
        dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        h  = dt.hour
        if 22 <= h or h < 8:
            return "ASIAN"
        if 8 <= h < 16:
            return "LONDON"
        return "NEW_YORK"
    except Exception:
        return "UNKNOWN"


def _load_trades(db_path: str) -> list[dict]:
    try:
        with _sqlite_connect(db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("""
                SELECT symbol, direction, regime, confidence, outcome, pnl, entry_time
                FROM trades
                WHERE status='CLOSED' AND outcome IS NOT NULL
                ORDER BY entry_time DESC
            """).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.debug(f"PatternMiner DB error: {e}")
        return []


def mine_patterns(db_path: str, min_trades: int = 10) -> dict:
    """
    Analyze trade history to produce a statistical edge map.

    Returns dict with keys:
        available:    bool
        total_trades: int
        by_symbol:    {sym  → {wins, total, win_rate}}
        by_regime:    {reg  → {wins, total, win_rate}}
        by_session:   {ses  → {wins, total, win_rate}}
        by_combo:     {"SYM|REG|SES" → {wins, total, win_rate}}
        conf_buckets: {"50-60" → {wins, total, win_rate}}
        top_setups:   list[str]
        weak_setups:  list[str]
        overall_rate: float
    """
    global _patterns_cache, _cache_trade_count

    trades = _load_trades(db_path)
    if not trades:
        return {"available": False, "total_trades": 0}

    total = len(trades)
    if total == _cache_trade_count and _patterns_cache.get("available"):
        return _patterns_cache

    by_symbol:  dict = defaultdict(lambda: {"wins": 0, "total": 0})
    by_regime:  dict = defaultdict(lambda: {"wins": 0, "total": 0})
    by_session: dict = defaultdict(lambda: {"wins": 0, "total": 0})
    by_combo:   dict = defaultdict(lambda: {"wins": 0, "total": 0})
    conf_b:     dict = defaultdict(lambda: {"wins": 0, "total": 0})

    for t in trades:
        sym     = (t.get("symbol") or "?").upper().split(".")[0]
        regime  = t.get("regime",  "?") or "?"
        conf    = int(t.get("confidence", 0) or 0)
        outcome = t.get("outcome", "") or ""
        session = _session_from_time(t.get("entry_time", "") or "")
        is_win  = outcome in ("WIN", "WIN_STRONG")
        bucket  = f"{(conf // 10) * 10}-{(conf // 10) * 10 + 10}"
        key     = f"{sym}|{regime}|{session}"

        for d, k in [(by_symbol, sym), (by_regime, regime), (by_session, session),
                     (by_combo, key), (conf_b, bucket)]:
            d[k]["total"] += 1
            if is_win:
                d[k]["wins"] += 1

    def with_rates(d):
        return {k: {**v, "win_rate": v["wins"] / v["total"] if v["total"] else 0.0}
                for k, v in d.items()}

    sym_r  = with_rates(by_symbol)
    reg_r  = with_rates(by_regime)
    ses_r  = with_rates(by_session)
    com_r  = with_rates(by_combo)
    con_r  = with_rates(conf_b)

    total_wins   = sum(v["wins"] for v in by_symbol.values())
    overall_rate = total_wins / total if total else 0.0

    qualified = sorted(
        [(k, v) for k, v in com_r.items() if v["total"] >= min_trades],
        key=lambda x: x[1]["win_rate"], reverse=True,
    )
    top_setups  = [f"{k}: {v['win_rate']:.0%} ({v['total']})" for k, v in qualified[:5]]
    weak_setups = [f"{k}: {v['win_rate']:.0%} ({v['total']})" for k, v in qualified[-5:]]

    result = {
        "available":    True,
        "total_trades": total,
        "by_symbol":    sym_r,
        "by_regime":    reg_r,
        "by_session":   ses_r,
        "by_combo":     com_r,
        "conf_buckets": con_r,
        "top_setups":   top_setups,
        "weak_setups":  weak_setups,
        "overall_rate": overall_rate,
    }

    _patterns_cache      = result
    _cache_trade_count   = total
    return result


def get_setup_boost(symbol: str, regime: str, session: str, patterns: dict,
                    min_trades: int = 8) -> int:
    """
    Return ±confidence points based on this setup's historical win rate vs overall.

    ≥ +15pp above overall → +4
    ≥ +8pp above overall  → +2
    ≤ −15pp below overall → −4
    ≤ −8pp below overall  → −2
    """
    if not patterns.get("available") or patterns["total_trades"] < 20:
        return 0

    overall_rate = patterns.get("overall_rate", 0.5)
    sym = symbol.upper().split(".")[0]
    key = f"{sym}|{regime}|{session}"

    combo = patterns.get("by_combo", {}).get(key)
    if not combo or combo["total"] < min_trades:
        return 0

    diff = combo["win_rate"] - overall_rate
    if diff >= 0.15:
        return 4
    if diff >= 0.08:
        return 2
    if diff <= -0.15:
        return -4
    if diff <= -0.08:
        return -2
    return 0
