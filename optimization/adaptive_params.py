"""
Adaptive Parameter Optimizer — Walk-Forward Confidence Tuning.

After every N closed trades, analyzes recent win rates and adjusts
MIN_CONFIDENCE_TO_TRADE dynamically. This is a "soft walk-forward":
rather than re-running a full backtest, it reads the live trade journal
and adjusts the single most impactful gate parameter.

Logic:
  - If recent win rate < 42%: raise threshold by 5 (be more selective)
  - If recent win rate > 62%: lower threshold by 3 (capture more edge)
  - Never go below BASE_CONFIDENCE (50) or above MAX_CONFIDENCE (68)

Why these bounds:
  - 50 = minimum acceptable (5-agent risk desk catches bad calls below this)
  - 68 = maximum practical (above this, too few trades to generate data)
  - +5/-3 asymmetry = raise faster than you lower (drawdown protection)

Usage:
  from optimization.adaptive_params import compute_adaptive_confidence
  new_threshold, reason = compute_adaptive_confidence(db_path, lookback=30, current=50)
"""

import sqlite3
import os
import logging

log = logging.getLogger("AdaptiveParams")

BASE_CONFIDENCE = 50   # absolute floor — below this, risk system is overloaded
MAX_CONFIDENCE  = 68   # absolute ceiling — above this, trades become too rare


def compute_adaptive_confidence(
    db_path: str,
    lookback_trades: int = 30,
    current_threshold: int = 50,
) -> tuple[int, str]:
    """
    Compute a new MIN_CONFIDENCE_TO_TRADE value based on recent trade performance.

    Args:
        db_path:           Path to trade_journal.db
        lookback_trades:   Number of recent closed trades to analyze
        current_threshold: The current value of MIN_CONFIDENCE_TO_TRADE

    Returns:
        (new_threshold, reason_string)
        new_threshold == current_threshold if no adjustment is needed.
    """
    if not os.path.exists(db_path):
        return current_threshold, "No journal — no adjustment"

    try:
        with sqlite3.connect(db_path) as con:
            rows = con.execute("""
                SELECT confidence, pnl
                FROM trades
                WHERE status='CLOSED' AND confidence IS NOT NULL AND pnl IS NOT NULL
                ORDER BY exit_time DESC
                LIMIT ?
            """, (lookback_trades,)).fetchall()
    except Exception as e:
        log.warning(f"AdaptiveParams DB read failed: {e}")
        return current_threshold, "DB error — no adjustment"

    if len(rows) < 20:
        return current_threshold, f"Only {len(rows)} trades — need 20 to adjust"

    # Evaluate trades that qualified at or above the current threshold
    qualifying = [(c, p) for c, p in rows if c >= current_threshold]
    if len(qualifying) < 10:
        return current_threshold, f"Only {len(qualifying)} qualifying trades — insufficient"

    wins     = sum(1 for _, p in qualifying if p > 0)
    win_rate = wins / len(qualifying)

    # Confidence band breakdown (for logging)
    bands = {
        "50-59": [(c, p) for c, p in rows if 50 <= c < 60],
        "60-69": [(c, p) for c, p in rows if 60 <= c < 70],
        "70+":   [(c, p) for c, p in rows if c >= 70],
    }
    band_summary = " | ".join(
        f"{band}: {sum(1 for _,p in trades if p>0)}/{len(trades)}"
        for band, trades in bands.items() if trades
    )

    if win_rate < 0.42:
        new_th = min(MAX_CONFIDENCE, current_threshold + 5)
        reason = (
            f"Recent win rate {win_rate:.1%} ({wins}/{len(qualifying)} trades) < 42% "
            f"→ raising confidence {current_threshold}→{new_th} | {band_summary}"
        )
        log.info(f"AdaptiveParams: {reason}")
        return new_th, reason

    if win_rate > 0.62:
        new_th = max(BASE_CONFIDENCE, current_threshold - 3)
        reason = (
            f"Recent win rate {win_rate:.1%} ({wins}/{len(qualifying)} trades) > 62% "
            f"→ lowering confidence {current_threshold}→{new_th} | {band_summary}"
        )
        log.info(f"AdaptiveParams: {reason}")
        return new_th, reason

    reason = (
        f"Win rate {win_rate:.1%} ({wins}/{len(qualifying)} trades) in 42–62% range "
        f"— no adjustment | {band_summary}"
    )
    log.debug(f"AdaptiveParams: {reason}")
    return current_threshold, reason


def analyze_by_confidence_band(db_path: str, lookback_trades: int = 50) -> dict:
    """
    Full confidence band analysis — shows which confidence level performs best.
    Used in the daily report and for manual tuning decisions.

    Returns dict keyed by band label.
    """
    result: dict[str, dict] = {}
    if not os.path.exists(db_path):
        return result

    try:
        with sqlite3.connect(db_path) as con:
            rows = con.execute("""
                SELECT confidence, pnl
                FROM trades
                WHERE status='CLOSED' AND confidence IS NOT NULL AND pnl IS NOT NULL
                ORDER BY exit_time DESC
                LIMIT ?
            """, (lookback_trades,)).fetchall()
    except Exception:
        return result

    bands = [
        ("45-49", 45, 50),
        ("50-54", 50, 55),
        ("55-59", 55, 60),
        ("60-64", 60, 65),
        ("65-69", 65, 70),
        ("70-74", 70, 75),
        ("75+",   75, 101),
    ]
    for label, lo, hi in bands:
        subset = [(c, p) for c, p in rows if lo <= c < hi]
        if not subset:
            continue
        wins     = sum(1 for _, p in subset if p > 0)
        total_pnl = sum(p for _, p in subset)
        result[label] = {
            "trades":   len(subset),
            "wins":     wins,
            "win_rate": round(wins / len(subset) * 100, 1),
            "total_pnl": round(total_pnl, 2),
        }

    return result
