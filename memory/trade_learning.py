"""
What happens after a trade closes — shared by bot.py and mcp_server.py, so a
trade teaches the bot the same things whichever of them opened it.

    classify_outcome      R-multiple → WIN_STRONG / WIN / BREAKEVEN / LOSS / STOP_LOSS
    update_blacklist      setup blacklist (symbol + regime) after a win or loss
    save_training_sample  ML gate training row + online (SGD) update
    lesson_fields         extra fields stored with a reflection lesson
    loss_streak           consecutive losses on a symbol, from the journal
    json_safe_context     entry context without the DataFrame, ready to store
"""

import json
import logging
import math

from strategy.indicators import feature_vector

log = logging.getLogger("TradeLearning")


def classify_outcome(r_multiple: float) -> str:
    """R = net_pnl / initial_risk_usd. 1.0R = made back exactly what was risked."""
    if r_multiple >= 1.5:   return "WIN_STRONG"   # 150%+ of risk recovered
    if r_multiple >= 0.3:   return "WIN"           # at least 30% of risk recovered
    if r_multiple >= -0.15: return "BREAKEVEN"     # flat within spread/commission
    if r_multiple >= -0.8:  return "LOSS"          # partial loss
    return "STOP_LOSS"                             # full or near-full stop hit


def regime_of(entry_context: dict) -> str:
    rc = (entry_context or {}).get("regime", {})
    return rc.get("regime", "") if isinstance(rc, dict) else str(rc or "")


def json_safe_context(ctx: dict) -> dict:
    """Entry context minus private keys (e.g. the DataFrame), as plain JSON types."""
    def conv(v):
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items() if not str(k).startswith("_")}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if hasattr(v, "item") and not isinstance(v, (str, bytes)):
            try:
                v = v.item()
            except Exception:
                pass
        if hasattr(v, "isoformat"):
            return v.isoformat()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)
    return json.loads(json.dumps(conv(ctx or {})))


def update_blacklist(journal, cfg, symbol: str, regime: str, is_win: bool) -> int:
    """Setup blacklist: a win resets symbol+regime, a loss adds one. Returns the count."""
    if not getattr(cfg, "SETUP_BLACKLIST_ENABLED", False):
        return 0
    if is_win:
        journal.reset_blacklist(symbol, regime)
        return 0
    count = journal.add_blacklist_loss(symbol, regime)
    if count >= getattr(cfg, "SETUP_BLACKLIST_THRESHOLD", 3):
        log.warning(f"[{symbol}] BLACKLISTED: {count} consecutive losses in {regime} regime "
                    f"— will skip next occurrence.")
    return count


def save_training_sample(memory, meta_gate, *, ticket, symbol, direction, entry_time,
                         exit_time, entry_context, r_multiple, pnl_pct, outcome,
                         is_win) -> bool:
    """
    One ML-gate training row (every closed trade, no usefulness filter) plus the
    online SGD update. Trades without an entry context have no features → skipped.
    """
    if not entry_context:
        return False
    fv = feature_vector(entry_context)
    entry_iso = entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time)
    exit_iso  = exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time)
    try:
        memory.save_ml_sample({
            "ticket": ticket, "symbol": symbol, "direction": direction,
            "entry_time": entry_iso, "exit_time": exit_iso, "feature_vector": fv,
            "regime": regime_of(entry_context),
            "fib_proximity": entry_context.get("fib_proximity", 0.0),
            "cot_z_score": entry_context.get("cot_z_score", 0.0),
            "econ_surprise": entry_context.get("econ_surprise", 0.0),
            "r_multiple": r_multiple, "pnl_pct": pnl_pct, "outcome": outcome,
        })
    except Exception as e:
        log.warning(f"[{symbol}] Could not store ML sample: {e}")
        return False
    if meta_gate is not None:
        try:
            meta_gate.online_update(
                feature_vector=fv, is_win=is_win, regime_str=regime_of(entry_context),
                fib_proximity=entry_context.get("fib_proximity", 0.0),
                cot_z_score=entry_context.get("cot_z_score", 0.0),
                econ_surprise=entry_context.get("econ_surprise", 0.0),
                direction=direction, entry_time=entry_time)
        except Exception as e:
            log.debug(f"Online update error: {e}")
    return True


def lesson_fields(entry_context: dict, r_multiple: float, entry_time) -> dict:
    """Fields added to a reflection lesson before memory.save()."""
    entry_iso = entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time)
    return {
        "feature_vector": feature_vector(entry_context),
        "r_multiple":     r_multiple,
        "regime":         regime_of(entry_context),
        "fib_proximity":  entry_context.get("fib_proximity", 0.0),
        "cot_z_score":    entry_context.get("cot_z_score", 0.0),
        "econ_surprise":  entry_context.get("econ_surprise", 0.0),
        "entry_time":     entry_iso,
        # this trade's training row lives in ml_samples → not counted twice
        "ml_sampled":     1 if entry_context else 0,
    }


def loss_streak(journal, symbol: str, lookback: int = 50) -> int:
    """Consecutive losing closed trades on `symbol`, most recent first (from the journal)."""
    streak = 0
    for row in journal.get_recent_trades(lookback):
        if row.get("symbol") != symbol or row.get("status") != "CLOSED":
            continue
        if (row.get("pnl") or 0) < 0:
            streak += 1
        else:
            break
    return streak
