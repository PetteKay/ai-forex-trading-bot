"""
Booking trades that closed without the booker watching them — by their stop
loss / take profit, by close_position, or while no process was running.

Used by BOTH bot.py (each tick, for trades it does not track: closed while no
bot ran) and mcp_server.py (while no bot is running). Only the holder of the
account lock may call it (bot_control.py, specs/BotControl.tla INV-BC-3), so a
trade is never booked twice.

Booking = journal outcome (the commit point: the row becomes CLOSED and is not
picked up again) · governor P&L (if closed today) · setup blacklist · ML-gate
training sample + online update · lesson · Telegram alert.
"""
from __future__ import annotations

import contextlib
import logging
from datetime import datetime, timezone

from memory.trade_learning import (classify_outcome, update_blacklist,
                                   save_training_sample, regime_of, lesson_fields)

log = logging.getLogger("TradeBooking")


def book_closed_trades(*, journal, memory, governor, meta_gate, mt5b, magic: int, cfg,
                       mt5_lock=None, exclude=(), telegram=None, llm_state=None,
                       lookback: int = 200) -> list[dict]:
    """
    Journal rows still OPEN whose ticket is not an open MT5 position (and not in
    `exclude`: trades the caller tracks and books itself) are booked once each.
    `meta_gate` may be a callable returning the gate (built only if needed).
    Returns one dict per booked trade.
    """
    lock = mt5_lock or contextlib.nullcontext()
    with lock:
        open_now = {p["ticket"] for p in mt5b.get_all_open_positions(magic)}
        equity = mt5b.get_account_equity()
    skip = set(open_now) | {int(t) for t in exclude}
    booked = []
    now = datetime.now(timezone.utc)
    today = now.date()
    for row in journal.get_recent_trades(lookback):
        if row.get("status") != "OPEN" or row["ticket"] in skip:
            continue
        ticket, sym = row["ticket"], row["symbol"]
        with lock:
            closed = mt5b.get_closed_trade(ticket, max_wait_secs=0)
        if not closed:
            continue          # history not available yet — try again next time
        net = float(closed.get("net_pnl", 0.0))
        exit_utc = None
        try:
            et = closed.get("exit_time")
            if et is not None:
                ts = et.replace(tzinfo=timezone.utc).timestamp() if et.tzinfo is None else et.timestamp()
                with lock:
                    exit_utc = mt5b.terminal_time_to_utc(int(ts), sym)
        except Exception:
            exit_utc = None
        if exit_utc is not None:
            counts_today = exit_utc.date() == today
        else:   # offset not measurable (market closed): fall back to the entry date
            counts_today = str(row.get("entry_time", ""))[:10] == today.isoformat()

        snap = memory.get_open_context(ticket) or {}
        ctx  = snap.get("context") or {}
        risk_usd   = float(snap.get("risk_usd") or 0.0)
        r_multiple = (net / risk_usd) if risk_usd > 0 else 0.0
        outcome    = classify_outcome(r_multiple)
        pnl_pct    = (net / equity) * 100 if equity else 0.0
        try:
            entry_dt = datetime.fromisoformat(snap.get("entry_time") or row.get("entry_time"))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        except Exception:
            entry_dt = now
        hold = max(0.0, (now - entry_dt).total_seconds() / 3600)
        direction = snap.get("direction") or row.get("direction")

        # commit point: from here on the row is CLOSED and never booked again
        journal.log_trade_close(ticket=ticket, exit_price=closed.get("exit_price", 0.0),
                                pnl=net, pnl_pct=pnl_pct, outcome=outcome,
                                hold_hours=round(hold, 2), exit_reason="SYNC")
        if counts_today:
            governor.record_trade_closed(net)
        update_blacklist(journal, cfg, sym, regime_of(ctx), net > 0)

        gate = meta_gate() if callable(meta_gate) else meta_gate
        sample = save_training_sample(
            memory, gate, ticket=ticket, symbol=sym, direction=direction,
            entry_time=entry_dt, exit_time=exit_utc or now, entry_context=ctx,
            r_multiple=r_multiple, pnl_pct=pnl_pct, outcome=outcome, is_win=net > 0)

        try:
            lesson = lesson_for(memory, cfg, ticket, sym, direction, snap, ctx, row, closed,
                                pnl_pct, r_multiple, outcome, hold,
                                llm_state if llm_state is not None else {}) \
                if ctx else "no entry snapshot"
        except Exception as e:
            lesson = f"lesson failed: {e}"
        memory.delete_open_context(ticket)
        if telegram is not None:
            try:
                telegram.trade_close(symbol=sym, direction=direction, outcome=outcome, pnl=net,
                                     r_multiple=r_multiple, hold_hours=hold)
            except Exception:
                pass
        booked.append({"ticket": ticket, "symbol": sym, "net_pnl": net,
                       "r_multiple": round(r_multiple, 2), "outcome": outcome,
                       "counted_in_today": counts_today, "ml_sample": bool(sample),
                       "lesson": lesson})
        log.info(f"Booked closed trade {ticket} {sym}: {net:+.2f} ({outcome}, R={r_multiple:+.2f})"
                 f"{'' if counts_today else ' (not today — journal only)'}")
    return booked


def lesson_for(memory, cfg, ticket, sym, direction, snap, ctx, row, closed,
               pnl_pct, r_multiple, outcome, hold, llm_state: dict) -> str:
    """Same selectivity as bot.py's reflection. Returns what happened.
    cfg.MCP_LESSONS: "client" queues it for the connected AI (get_trades_to_review
    → save_lesson); "llm" runs the bot's reflection agent (needs LLM_API_KEY)."""
    from agents import reflection
    pm   = snap.get("pm") or {}
    tech = snap.get("tech") or {}
    if not reflection._pre_filter(r_multiple, int(pm.get("confidence", 0) or 0), outcome):
        return "nothing to learn (below the reflection filter)"
    mode = str(getattr(cfg, "MCP_LESSONS", "client")).lower()
    if mode == "llm" and getattr(cfg, "LLM_API_KEY", ""):
        from agents import base as llm
        if not llm_state.get("llm_ready"):
            llm.init(cfg)
            llm_state["llm_ready"] = True
        rec = reflection.reflect(symbol=sym, direction=direction,
                                 entry_price=float(snap.get("entry_price") or row.get("entry_price") or 0),
                                 exit_price=float(closed.get("exit_price", 0.0)), pnl_pct=pnl_pct,
                                 r_multiple=r_multiple, hold_hours=hold, outcome=outcome,
                                 entry_context=ctx, technical_analysis=tech, pm_decision=pm, cfg=cfg)
        if not rec:
            return "reflection agent: nothing worth saving"
        rec.update(lesson_fields(ctx, r_multiple, snap.get("entry_time")))
        return "saved by reflection agent" if memory.save(rec) else "below usefulness threshold"
    memory.queue_review(ticket, {
        "symbol": sym, "direction": direction,
        "entry_price": snap.get("entry_price") or row.get("entry_price"),
        "exit_price": closed.get("exit_price"), "pnl_pct": round(pnl_pct, 3),
        "r_multiple": round(r_multiple, 2), "outcome": outcome, "hold_hours": round(hold, 1),
        "entry_time": snap.get("entry_time"), "reason_at_entry": snap.get("reason", ""),
        "source": snap.get("source", ""), "entry_context": ctx, "tech": tech, "pm": pm,
    })
    return "queued for review — call get_trades_to_review, then save_lesson"
