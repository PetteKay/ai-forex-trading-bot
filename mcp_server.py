"""
MCP server — lets Claude (Claude Code / Claude Desktop), OpenAI Codex, ChatGPT
or any MCP client drive the bot. The client's own model does the reasoning, so
this mode needs no LLM API key; every order still goes through the bot's risk
governor and gates.

Runs on the Windows PC that runs MetaTrader 5 (the MetaTrader5 Python package
is Windows-only).

  Local (Claude Code, Claude Desktop, Codex):   py mcp_server.py
  Remote (ChatGPT needs an HTTPS URL):           py mcp_server.py --http --port 8765
      → put it behind an HTTPS tunnel, and set MCP_HTTP_TOKEN (see MCP.md).

Trading is controlled by config.MCP_TRADING_MODE:
  "off"      no trading tools work
  "dry_run"  (default) place_trade only previews — no order is sent
  "live"     orders are sent to the logged-in MT5 account
Over HTTP, trading also needs config.MCP_HTTP_ALLOW_TRADING = True.

The autonomous bot (bot.py) and this server should not trade the same account
at the same time: both update observability/governor_state.json, and a running
bot does not manage positions this server opens until it restarts.
"""

import argparse
import hmac
import json
import logging
import math
import os
import sys
import threading
import types
from urllib.parse import parse_qs

# stdio transport: stdout carries the protocol, so ALL logging goes to stderr.
logging.basicConfig(
    level=logging.INFO, stream=sys.stderr,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("MCPServer")

import config                                        # noqa: E402
import mt5_bridge as mt5b                            # noqa: E402
from mcp.server.mcpserver import MCPServer           # noqa: E402
from mcp.types import ToolAnnotations                # noqa: E402
from risk_governor import RiskGovernor               # noqa: E402
from strategy.indicators import compute, feature_vector, market_fingerprint  # noqa: E402
from strategy.regime import classify as classify_regime                     # noqa: E402
from strategy.position_sizing import compute_lot                            # noqa: E402
from strategy.correlation import _check_correlation                         # noqa: E402

READ_ONLY   = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True)

server = MCPServer(
    name="ai-forex-trading-bot",
    instructions=(
        "Forex/CFD trading tools for a MetaTrader 5 account. Use analyze_market, get_news, "
        "get_ml_probability and recall_lessons to form a view; preview_trade before place_trade. "
        "Every order passes the bot's risk governor (daily loss cap, max positions, drawdown "
        "halt), duplicate and correlation guards. Trading is dry-run unless the owner enabled "
        "live mode. Nothing here is financial advice."
    ),
)

# ── Shared state ─────────────────────────────────────────────────────────────
_MT5_LOCK  = threading.RLock()     # the MT5 package is not documented as thread-safe
_TRADE_LOCK = threading.Lock()     # check-then-place is atomic: no two orders race the limits
_STATE     = {"connected": False, "transport": "stdio"}
_lazy: dict = {}


def _mt5_ready() -> None:
    if not _STATE["connected"]:
        if not mt5b.connect():
            raise RuntimeError("Cannot connect to MetaTrader 5. Is the terminal open and logged in?")
        _STATE["connected"] = True


def _jsonable(v):
    """Convert numpy / pandas values into plain JSON types."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items() if not str(k).startswith("_")}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
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


def _magic() -> int:
    return int(getattr(config, "MAGIC_NUMBER", 0))


def _trading_mode() -> str:
    return str(getattr(config, "MCP_TRADING_MODE", "dry_run")).lower()


def _trading_block_reason() -> str | None:
    mode = _trading_mode()
    if mode == "off":
        return "Trading tools are off (config.MCP_TRADING_MODE = 'off')."
    if _STATE["transport"] == "http" and not getattr(config, "MCP_HTTP_ALLOW_TRADING", False):
        return ("Trading over HTTP is disabled (config.MCP_HTTP_ALLOW_TRADING = False). "
                "Use the local stdio server, or enable it deliberately.")
    return None


def _market(symbol: str) -> tuple[str, dict]:
    """Closed-candle context for a symbol (same calculation as the bot)."""
    with _MT5_LOCK:
        _mt5_ready()
        sym = mt5b.resolve_symbol(symbol.upper())
        df  = mt5b.get_candles(sym, config.TIMEFRAME, count=250)
        htf = mt5b.get_candles(sym, config.HTF_TIMEFRAME, count=100)
        live = mt5b.get_live_price(sym)
        spread = mt5b.get_spread_pips(sym)
    ctx = compute(df, config, htf_df=htf, symbol=sym)
    if not ctx:
        raise RuntimeError(f"Not enough candle data for {sym}.")
    ctx["regime"]      = classify_regime(ctx["_df"], ctx["atr_percentile"])
    ctx["live_price"]  = live or ctx["price"]
    ctx["spread_pips"] = spread
    ctx["last_closed_candle"] = df.index[-1]
    return sym, ctx


def _governor(equity: float) -> RiskGovernor:
    """Fresh governor from the shared state file, so restarts and the bot's limits apply."""
    return RiskGovernor(
        starting_equity    = equity,
        daily_loss_cap_pct = config.DAILY_LOSS_CAP_PCT,
        max_open_positions = config.MAX_OPEN_POSITIONS,
        max_orders_per_day = config.MAX_ORDERS_PER_DAY,
        max_drawdown_pct   = config.MAX_DRAWDOWN_PCT,
        allowed_sessions   = config.ALLOWED_SESSIONS,
        state_path         = getattr(config, "GOVERNOR_STATE_PATH", "observability/governor_state.json"),
        account_id         = mt5b.get_account_login(),
    )


def _book_closed_trades(gov: RiskGovernor) -> list[dict]:
    """
    Book trades that have closed since the last check — closed by close_position,
    by their stop loss / take profit, or while the server was not running — and
    learn from them exactly as bot.py does (memory/trade_learning.py):

      daily loss cap (governor) · journal · setup blacklist · ML-gate training
      sample + online update · periodic retrain · Telegram alert · lesson

    Source: journal rows still OPEN whose ticket is no longer an open MT5
    position. Each is booked once (the journal row is then CLOSED). Its P&L
    counts toward today's limit only if the trade closed today (UTC).
    """
    from datetime import datetime, timezone
    from agents import reflection
    from memory.trade_learning import (classify_outcome, update_blacklist,
                                       save_training_sample, regime_of)
    journal, memory = _journal(), _memory()
    with _MT5_LOCK:
        open_now = {p["ticket"] for p in mt5b.get_all_open_positions(_magic())}
        equity = mt5b.get_account_equity()
    booked = []
    now = datetime.now(timezone.utc)
    today = now.date()
    for row in journal.get_recent_trades(200):
        if row.get("status") != "OPEN" or row["ticket"] in open_now:
            continue
        ticket, sym = row["ticket"], row["symbol"]
        with _MT5_LOCK:
            closed = mt5b.get_closed_trade(ticket, max_wait_secs=0)
        if not closed:
            continue          # history not available yet — try again next time
        net = float(closed.get("net_pnl", 0.0))
        exit_utc = None
        try:
            et = closed.get("exit_time")
            if et is not None:
                ts = et.replace(tzinfo=timezone.utc).timestamp() if et.tzinfo is None else et.timestamp()
                with _MT5_LOCK:
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
        except Exception:
            entry_dt = now
        hold = (now - entry_dt).total_seconds() / 3600
        direction = snap.get("direction") or row.get("direction")

        journal.log_trade_close(ticket=ticket, exit_price=closed.get("exit_price", 0.0),
                                pnl=net, pnl_pct=pnl_pct, outcome=outcome,
                                hold_hours=round(hold, 2), exit_reason="MCP_SYNC")
        if counts_today:
            gov.record_trade_closed(net)
        update_blacklist(journal, config, sym, regime_of(ctx), net > 0)

        sample = save_training_sample(
            memory, _meta_gate(), ticket=ticket, symbol=sym, direction=direction,
            entry_time=entry_dt, exit_time=exit_utc or now, entry_context=ctx,
            r_multiple=r_multiple, pnl_pct=pnl_pct, outcome=outcome, is_win=net > 0)
        if sample:
            _lazy["since_retrain"] = _lazy.get("since_retrain", 0) + 1
            if _lazy["since_retrain"] >= getattr(config, "META_RETRAIN_EVERY", 20):
                _meta_gate().train(memory.get_all_for_training(),
                                   training_lookback_days=getattr(config, "ML_TRAINING_LOOKBACK_DAYS", 180))
                _lazy["since_retrain"] = 0

        lesson = _lesson_for(reflection, ticket, sym, direction, snap, ctx, row, closed,
                             pnl_pct, r_multiple, outcome, hold) if ctx else "no entry snapshot"
        memory.delete_open_context(ticket)
        try:
            _telegram().trade_close(symbol=sym, direction=direction, outcome=outcome, pnl=net,
                                    r_multiple=r_multiple, hold_hours=hold)
        except Exception:
            pass
        booked.append({"ticket": ticket, "symbol": sym, "net_pnl": net, "r_multiple": round(r_multiple, 2),
                       "outcome": outcome, "counted_in_today": counts_today,
                       "ml_sample": bool(sample), "lesson": lesson})
        log.info(f"Booked closed trade {ticket} {sym}: {net:+.2f} ({outcome}, R={r_multiple:+.2f})"
                 f"{'' if counts_today else ' (not today — journal only)'}")
    return booked


def _lesson_for(reflection, ticket, sym, direction, snap, ctx, row, closed,
                pnl_pct, r_multiple, outcome, hold) -> str:
    """Same selectivity as bot.py's reflection. Returns what happened."""
    pm   = snap.get("pm") or {}
    tech = snap.get("tech") or {}
    if not reflection._pre_filter(r_multiple, int(pm.get("confidence", 0) or 0), outcome):
        return "nothing to learn (below the reflection filter)"
    mode = str(getattr(config, "MCP_LESSONS", "client")).lower()
    if mode == "llm" and getattr(config, "LLM_API_KEY", ""):
        from agents import base as llm
        from memory.trade_learning import lesson_fields
        if not _lazy.get("llm_ready"):
            llm.init(config)
            _lazy["llm_ready"] = True
        rec = reflection.reflect(symbol=sym, direction=direction,
                                 entry_price=float(snap.get("entry_price") or row.get("entry_price") or 0),
                                 exit_price=float(closed.get("exit_price", 0.0)), pnl_pct=pnl_pct,
                                 r_multiple=r_multiple, hold_hours=hold, outcome=outcome,
                                 entry_context=ctx, technical_analysis=tech, pm_decision=pm, cfg=config)
        if not rec:
            return "reflection agent: nothing worth saving"
        rec.update(lesson_fields(ctx, r_multiple, snap.get("entry_time")))
        return "saved by reflection agent" if _memory().save(rec) else "below usefulness threshold"
    _memory().queue_review(ticket, {
        "symbol": sym, "direction": direction,
        "entry_price": snap.get("entry_price") or row.get("entry_price"),
        "exit_price": closed.get("exit_price"), "pnl_pct": round(pnl_pct, 3),
        "r_multiple": round(r_multiple, 2), "outcome": outcome, "hold_hours": round(hold, 1),
        "entry_time": snap.get("entry_time"), "reason_at_entry": snap.get("reason", ""),
        "source": snap.get("source", ""), "entry_context": ctx, "tech": tech, "pm": pm,
    })
    return "queued for review — call get_trades_to_review, then save_lesson"


def _get(name, factory):
    if name not in _lazy:
        _lazy[name] = factory()
    return _lazy[name]


def _telegram():
    from notifications.telegram_alert import TelegramAlert
    return _get("telegram", lambda: TelegramAlert(
        token=getattr(config, "TELEGRAM_BOT_TOKEN", ""),
        chat_id=getattr(config, "TELEGRAM_CHAT_ID", ""),
        enabled=getattr(config, "TELEGRAM_ALERTS_ENABLED", False)))


def _memory():
    from memory.store import TradeMemoryStore
    return _get("memory", lambda: TradeMemoryStore(
        config.MEMORY_DB_PATH, config.MEMORY_USEFULNESS_THRESHOLD, config.MEMORY_MAX_STORED))


def _journal():
    from observability.journal import Journal
    return _get("journal", lambda: Journal(config.JOURNAL_DB_PATH))


def _calendar():
    from data.calendar import EconomicCalendar
    return _get("calendar", lambda: EconomicCalendar(
        blackout_before_min=config.CALENDAR_BLACKOUT_BEFORE_MIN,
        blackout_after_min=config.CALENDAR_BLACKOUT_AFTER_MIN,
        csv_path=config.CALENDAR_CSV_PATH,
        live_enabled=getattr(config, "LIVE_CALENDAR_ENABLED", True)))


def _meta_gate():
    def build():
        from ml.meta_label import MetaLabelGate
        g = MetaLabelGate()
        g.train(_memory().get_all_for_training(),
                training_lookback_days=getattr(config, "ML_TRAINING_LOOKBACK_DAYS", 180))
        return g
    return _get("meta_gate", build)


# ── Read-only tools ──────────────────────────────────────────────────────────

@server.tool(annotations=READ_ONLY)
def get_account_status() -> dict:
    """Account equity and balance, the bot's open positions, the risk governor's
    state (daily P&L, kill switch, drawdown halt) and the trading mode."""
    with _MT5_LOCK:
        _mt5_ready()
        equity  = mt5b.get_account_equity()
        balance = mt5b.get_account_balance()
        positions = mt5b.get_all_open_positions(_magic())
    gov = _governor(equity)
    booked = _book_closed_trades(gov)
    return _jsonable({
        "equity": equity, "balance": balance,
        "newly_booked_closed_trades": booked,
        "open_positions": positions,
        "governor": gov.status,
        "limits": {"daily_loss_cap_pct": config.DAILY_LOSS_CAP_PCT,
                   "max_open_positions": config.MAX_OPEN_POSITIONS,
                   "max_orders_per_day": config.MAX_ORDERS_PER_DAY,
                   "max_drawdown_pct": config.MAX_DRAWDOWN_PCT},
        "trading_mode": _trading_mode(),
        "trading_blocked": _trading_block_reason(),
        "sizing_mode": getattr(config, "SIZING_MODE", "fixed"),
    })


@server.tool(annotations=READ_ONLY)
def analyze_market(symbol: str) -> dict:
    """Technical picture for one symbol from CLOSED candles on the bot's timeframe
    (config.TIMEFRAME, with config.HTF_TIMEFRAME for higher-timeframe bias):
    indicators, market structure, candle patterns, regime (ADX/Hurst), ATR,
    sessions, live bid and spread. Example symbol: "EURUSD"."""
    sym, ctx = _market(symbol)
    out = {k: v for k, v in ctx.items() if not k.startswith("_")}
    out["symbol"] = sym
    out["timeframe"] = config.TIMEFRAME
    out["htf_timeframe"] = config.HTF_TIMEFRAME
    return _jsonable(out)


@server.tool(annotations=READ_ONLY)
def get_news(symbol: str) -> dict:
    """Recent headlines ranked for the symbol, live macro context (VIX, DXY,
    yields, gold, fear & greed) and the next high-impact economic event."""
    from data.news_feed import fetch_headlines
    out = {"symbol": symbol.upper(), "headlines": fetch_headlines(symbol.upper())}
    try:
        from data.macro_context import get_macro_context
        out["macro"] = get_macro_context(symbol.upper())
    except Exception as e:
        out["macro"] = {"error": str(e)}
    cal = _calendar()
    blackout, why = cal.is_blackout()
    out["calendar_blackout"] = {"active": blackout, "reason": why}
    out["next_event"] = cal.next_event()
    return _jsonable(out)


@server.tool(annotations=READ_ONLY)
def get_ml_probability(symbol: str, direction: str) -> dict:
    """The bot's meta-label model: probability that a BUY or SELL on this symbol
    right now wins, from its trade history. Below config.META_LABEL_MIN_PROB the
    bot would not take the trade. Uses a heuristic until enough trades exist."""
    direction = direction.upper()
    if direction not in ("BUY", "SELL"):
        raise ValueError("direction must be BUY or SELL")
    sym, ctx = _market(symbol)
    res = _meta_gate().predict(feature_vector(ctx), ctx, direction=direction)
    res["symbol"], res["direction"] = sym, direction
    res["min_probability_to_trade"] = config.META_LABEL_MIN_PROB
    return _jsonable(res)


@server.tool(annotations=READ_ONLY)
def recall_lessons(symbol: str, count: int = 5) -> dict:
    """Lessons the bot saved from similar past trades on this symbol (wins and losses)."""
    sym, ctx = _market(symbol)
    mems = _memory().recall_balanced(market_fingerprint(ctx), sym, max(1, min(count, 10)),
                                     feature_vector(ctx), min_losses=2)
    return _jsonable({"symbol": sym, "lessons": [
        {"outcome": m.get("outcome"), "direction": m.get("direction"),
         "lesson": m.get("lesson"), "r_multiple": m.get("r_multiple")} for m in mems]})


@server.tool(annotations=READ_ONLY)
def get_recent_trades(count: int = 20) -> dict:
    """The bot's most recent trades from its journal (open and closed)."""
    return _jsonable({"trades": _journal().get_recent_trades(max(1, min(count, 100)))})


# ── Trading tools ────────────────────────────────────────────────────────────

def _plan(symbol: str, direction: str, size_multiplier: float) -> dict:
    direction = direction.upper()
    if direction not in ("BUY", "SELL"):
        raise ValueError("direction must be BUY or SELL")
    if not (0.0 < size_multiplier <= 1.0):
        raise ValueError("size_multiplier must be between 0 and 1 (1 = full size)")
    sym, ctx = _market(symbol)
    with _MT5_LOCK:
        equity    = mt5b.get_account_equity()
        positions = mt5b.get_all_open_positions(_magic())
        import MetaTrader5 as mt5
        info = mt5.symbol_info(sym)
    blockers = []

    gov = _governor(equity)
    _book_closed_trades(gov)          # realised losses count before approving anything
    blackout, why = _calendar().is_blackout()
    ok, reason = gov.approve(current_equity=equity, open_position_count=len(positions),
                             sessions=ctx.get("sessions", []), is_blackout=blackout)
    if not ok:
        blockers.append(f"Risk governor: {reason}")
    if blackout:
        blockers.append(f"Economic calendar blackout: {why}")
    if any(p["symbol"] == sym and p["type"] == direction for p in positions):
        blockers.append(f"Already have a {direction} position on {sym}.")
    peers = {p["ticket"]: types.SimpleNamespace(symbol=p["symbol"], direction=p["type"])
             for p in positions}
    corr = _check_correlation(sym, direction, peers)
    if corr:
        blockers.append(f"Correlation guard: {corr} already open {direction}.")
    if getattr(config, "SPREAD_FILTER_ENABLED", False):
        max_sp = config.MAX_SPREAD_PIPS.get(sym.upper().split(".")[0],
                                            config.MAX_SPREAD_PIPS.get("DEFAULT", 5.0))
        if ctx["spread_pips"] and ctx["spread_pips"] > max_sp:
            blockers.append(f"Spread {ctx['spread_pips']} pips > max {max_sp}.")

    sl_mult, tp_mult = config.ATR_SL_MULT, config.ATR_TP_MULT
    try:
        if getattr(config, "SYMBOL_PARAMS_ENABLED", False):
            from strategy.symbol_params import get as get_symbol_params
            sp = get_symbol_params(sym)
            sl_mult, tp_mult = sp.get("ATR_SL_MULT", sl_mult), sp.get("ATR_TP_MULT", tp_mult)
    except Exception:
        pass
    # ── What the bot has learned (same rules as bot.py) ──────────────────
    from memory.trade_learning import loss_streak
    from strategy.position_sizing import kelly_fraction
    adjustments = []
    regime_name = ctx["regime"].get("regime", "")
    if getattr(config, "SETUP_BLACKLIST_ENABLED", False):
        th = getattr(config, "SETUP_BLACKLIST_THRESHOLD", 3)
        if _journal().is_blacklisted(sym, regime_name, th):
            blockers.append(f"Setup blacklist: {th}+ consecutive losses on {sym} in "
                            f"{regime_name} (cleared by the next win).")
    ml = _meta_gate().predict(feature_vector(ctx), ctx, direction=direction)
    min_p = config.META_LABEL_MIN_PROB
    if ml["probability"] < min_p:
        blockers.append(f"ML gate: win probability {ml['probability']:.3f} < {min_p} "
                        f"({'model' if ml['use_ml'] else 'heuristic'}).")
    if getattr(config, "USE_KELLY_SIZING", True) and ml.get("use_ml"):
        frac = kelly_fraction(ml["probability"], tp_mult / sl_mult if sl_mult > 0 else 0,
                              getattr(config, "KELLY_FRACTION", 0.25))
        if frac <= 0:
            blockers.append("Kelly: no edge at these odds.")
        elif frac < size_multiplier:
            adjustments.append(f"Kelly {size_multiplier:.2f} → {frac:.2f}")
            size_multiplier = frac
    if getattr(config, "LOSS_STREAK_ENABLED", False):
        streak = loss_streak(_journal(), sym)
        if streak >= getattr(config, "LOSS_STREAK_THRESHOLD", 3):
            m = getattr(config, "LOSS_STREAK_SIZE_MULT", 0.5)
            adjustments.append(f"Loss streak {streak} → size × {m}")
            size_multiplier *= m
    if getattr(config, "VOL_ADJUSTED_SIZING", False):
        atr_p = ctx.get("atr_percentile", 50)
        if atr_p >= getattr(config, "VOL_SIZING_EXTREME_ATR_PCT", 92):
            m = getattr(config, "VOL_SIZING_EXTREME_ATR_MULT", 0.50)
            adjustments.append(f"Extreme volatility (ATR {atr_p:.0f}th pct) → size × {m}")
            size_multiplier *= m
        elif atr_p >= getattr(config, "VOL_SIZING_HIGH_ATR_PCT", 80):
            m = getattr(config, "VOL_SIZING_HIGH_ATR_MULT", 0.70)
            adjustments.append(f"High volatility (ATR {atr_p:.0f}th pct) → size × {m}")
            size_multiplier *= m

    point     = float(getattr(info, "point", 0.00001) or 0.00001)
    sl_points = round(ctx["atr"] / point * sl_mult, 1)
    tp_points = round(ctx["atr"] / point * tp_mult, 1)
    tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
    tick_size  = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
    lot, lot_note = compute_lot(
        getattr(config, "SIZING_MODE", "fixed"), size_multiplier,
        base_lot=config.LOT_SIZE, equity=equity,
        risk_pct=getattr(config, "RISK_PER_TRADE_PCT", 0.5),
        sl_distance=sl_points * point, tick_value=tick_value, tick_size=tick_size,
        vol_min=float(getattr(info, "volume_min", 0.01) or 0.01),
        vol_step=float(getattr(info, "volume_step", 0.01) or 0.01),
        vol_max=float(getattr(info, "volume_max", 100.0) or 100.0),
        max_lot=getattr(config, "MAX_LOT_SIZE", None))
    if lot <= 0:
        blockers.append(f"Sizing: {lot_note}")
    price = ctx["live_price"]
    sign  = 1 if direction == "BUY" else -1
    risk_usd = (lot * (sl_points * point) / tick_size * tick_value) if tick_size else None
    return {
        "symbol": sym, "direction": direction, "allowed": not blockers, "blockers": blockers,
        "lot": lot, "sizing": lot_note, "approx_entry": price,
        "stop_loss": round(price - sign * sl_points * point, 6),
        "take_profit": round(price + sign * tp_points * point, 6),
        "sl_points": sl_points, "tp_points": tp_points,
        "risk_usd_if_stopped": round(risk_usd, 2) if risk_usd is not None else None,
        "regime": ctx["regime"].get("regime"),
        "ml_gate": {"probability": ml["probability"], "min": min_p, "model": ml.get("model_type"),
                    "trained_on": ml.get("n_samples", 0)},
        "size_multiplier": round(size_multiplier, 4), "size_adjustments": adjustments,
        "_ctx": ctx, "_gov": gov,
    }


@server.tool(annotations=READ_ONLY)
def preview_trade(symbol: str, direction: str, size_multiplier: float = 1.0) -> dict:
    """Check a BUY or SELL against every rule the bot uses — risk governor, calendar
    blackout, duplicate and correlation guards, spread, setup blacklist, ML gate,
    Kelly, loss-streak and volatility sizing — and show the lot, stop loss, take
    profit and money at risk. Sends no order."""
    plan = _plan(symbol, direction, size_multiplier)
    return _jsonable(plan)


@server.tool(annotations=DESTRUCTIVE)
def place_trade(symbol: str, direction: str, size_multiplier: float = 1.0,
                reason: str = "", confidence: int = 0) -> dict:
    """Open a market BUY or SELL with the bot's ATR stop loss and take profit.
    Runs every check in preview_trade first and refuses if any fails. In
    config.MCP_TRADING_MODE = "dry_run" (the default) it only reports what it
    would do. `reason`: why you are taking the trade (saved, and shown again when
    the trade is reviewed). `confidence`: your confidence 0-100 (used, like the
    bot's Fund Manager confidence, to decide which trades are worth a lesson)."""
    blocked = _trading_block_reason()
    if blocked:
        return {"placed": False, "error": blocked}
    with _TRADE_LOCK:
        return _place_locked(symbol, direction, size_multiplier, reason, confidence)


def _place_locked(symbol: str, direction: str, size_multiplier: float, reason: str,
                  confidence: int = 0) -> dict:
    plan = _plan(symbol, direction, size_multiplier)
    ctx, gov = plan["_ctx"], plan["_gov"]
    if not plan["allowed"]:
        return _jsonable({"placed": False, **plan})
    if _trading_mode() != "live":
        return _jsonable({"placed": False, "dry_run": True,
                          "note": "Dry run: config.MCP_TRADING_MODE is not 'live'. No order sent.",
                          **plan})
    with _MT5_LOCK:
        result = mt5b.place_order(
            symbol=plan["symbol"], direction=plan["direction"], lot_size=plan["lot"],
            sl_pips=plan["sl_points"], tp_pips=plan["tp_points"], magic=_magic(),
            comment="AIBot|MCP")
    if not result:
        return _jsonable({"placed": False, "error": "MT5 rejected the order (see server log).", **plan})
    gov.record_order_placed()
    confidence = max(0, min(100, int(confidence or 0)))
    try:
        _journal().log_trade_open(
            ticket=result["ticket"], symbol=plan["symbol"], direction=plan["direction"],
            entry_price=result["price"], sl=result["sl"], tp=result["tp"],
            volume=plan["lot"], confidence=confidence, regime=plan["regime"] or "",
            fingerprint=market_fingerprint(ctx), size_multiplier=plan.get("size_multiplier", 1.0))
    except Exception as e:
        log.warning(f"Journal write failed: {e}")
    # Entry snapshot — what the lesson and the ML training sample are built from
    try:
        from datetime import datetime, timezone
        from memory.trade_learning import json_safe_context
        risk_usd = 0.0
        try:
            with _MT5_LOCK:
                risk_usd = mt5b.get_risk_usd(plan["symbol"], result["price"], result["sl"], plan["lot"])
        except Exception as e:      # the plan's own risk figure is the fallback
            log.warning(f"MT5 risk calc failed, using the plan's risk: {e}")
        _memory().save_open_context(
            result["ticket"], symbol=plan["symbol"], direction=plan["direction"],
            entry_time=datetime.now(timezone.utc).isoformat(), entry_price=result["price"],
            risk_usd=risk_usd or plan.get("risk_usd_if_stopped") or 0.0,
            context=json_safe_context(ctx),
            tech={"summary": f"MCP client analysis: {reason}"[:500]},
            pm={"action": plan["direction"], "confidence": confidence, "reasoning": reason[:1000]},
            source="mcp", reason=reason[:1000])
    except Exception as e:
        log.warning(f"Entry snapshot not stored (no lesson/ML sample for this trade): {e}", exc_info=True)
    try:
        _telegram().trade_open(symbol=plan["symbol"], direction=plan["direction"],
                               entry_price=result["price"], sl=result["sl"], tp=result["tp"],
                               lot=plan["lot"], confidence=confidence,
                               regime=plan["regime"] or "", reasoning=reason[:120])
    except Exception:
        pass
    log.warning(f"MCP order placed: {plan['direction']} {plan['lot']} {plan['symbol']} "
                f"ticket={result['ticket']} reason={reason[:120]!r}")
    return _jsonable({"placed": True, "ticket": result["ticket"], "entry": result["price"],
                      "stop_loss": result["sl"], "take_profit": result["tp"],
                      "lot": plan["lot"], "symbol": plan["symbol"], "direction": plan["direction"]})


@server.tool(annotations=DESTRUCTIVE)
def close_position(ticket: int) -> dict:
    """Close one of the bot's open positions by ticket (only positions with the
    bot's magic number). Dry run unless config.MCP_TRADING_MODE = "live"."""
    blocked = _trading_block_reason()
    if blocked:
        return {"closed": False, "error": blocked}
    with _MT5_LOCK:
        _mt5_ready()
        pos = next((p for p in mt5b.get_all_open_positions(_magic()) if p["ticket"] == ticket), None)
        if pos is None:
            return {"closed": False, "error": f"No open bot position with ticket {ticket}."}
        if _trading_mode() != "live":
            return _jsonable({"closed": False, "dry_run": True, "position": pos,
                              "note": "Dry run: config.MCP_TRADING_MODE is not 'live'."})
        ok = mt5b.close_position(ticket, pos["symbol"], pos["volume"], pos["type"], magic=_magic())
        equity = mt5b.get_account_equity()
    booked = _book_closed_trades(_governor(equity)) if ok else []
    return _jsonable({"closed": bool(ok), "position": pos, "booked": booked})


# ── Lessons (the connected AI does the bot's reflection) ──────────────────

_LESSON_CATEGORIES = ("ENTRY_TIMING", "EXIT_TIMING", "REGIME_FILTER", "RISK_MANAGEMENT",
                      "VOLATILITY", "SESSION")
_LESSON_GUIDE = (
    "You are the bot's trading coach. Be extremely selective: save a lesson only if it is "
    "specific enough that a trader reading it knows exactly what to do differently. Never "
    "save generic advice ('wait for confirmation', 'follow the trend'). Usefulness: 0.8-1.0 "
    "counterintuitive result, critical failure or a setup that clearly works/fails under "
    "specific conditions; 0.6-0.8 clear lesson that should change behaviour; 0.4-0.6 mild, "
    "save only if new; below 0.5 not worth saving. Lessons below the bot's usefulness "
    "threshold are not stored. Categories: " + ", ".join(_LESSON_CATEGORIES) + "."
)
_REVIEW_KEYS = ("market_structure", "sessions", "rsi", "rsi_state", "ema_cross_up",
                "ema_cross_down", "fast_above_slow", "price_above_trend", "macd_positive",
                "bb_pct", "atr_percentile", "bb_squeeze", "regime", "candlestick_patterns")


@server.tool(annotations=READ_ONLY)
def get_trades_to_review(count: int = 5) -> dict:
    """Closed trades waiting for a lesson (the review the bot's reflection agent does).
    For each: the result, the market state at entry and the reason given at entry.
    Write the lesson with save_lesson, or call it with should_save=false."""
    booked = _book_closed_trades(_governor(mt5b_equity()))
    out = []
    for r in _memory().pending_reviews(max(1, min(count, 20))):
        ctx = r.pop("entry_context", {}) or {}
        r["market_state_at_entry"] = {k: ctx.get(k) for k in _REVIEW_KEYS if k in ctx}
        r.pop("tech", None)
        out.append(r)
    return _jsonable({"guide": _LESSON_GUIDE, "trades": out, "newly_booked": booked,
                      "usefulness_threshold": config.MEMORY_USEFULNESS_THRESHOLD})


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                         idempotent_hint=True, open_world_hint=False))
def save_lesson(ticket: int, lesson: str, usefulness_score: float, category: str,
                pattern_name: str = "", should_save: bool = True) -> dict:
    """Save the lesson from a reviewed trade into the bot's memory — recalled before
    future trades by bot.py and by recall_lessons. usefulness_score 0-1; category one
    of ENTRY_TIMING, EXIT_TIMING, REGIME_FILTER, RISK_MANAGEMENT, VOLATILITY, SESSION.
    Use should_save=false when the trade taught nothing new."""
    from memory.trade_learning import lesson_fields
    from strategy.indicators import market_fingerprint as fingerprint
    mem = _memory()
    rev = mem.get_review(ticket)
    if rev is None:
        return {"saved": False, "error": f"Trade {ticket} is not waiting for a review."}
    if rev.get("status") != "PENDING":
        return {"saved": False, "error": f"Trade {ticket} was already reviewed ({rev['status']})."}
    if not should_save:
        mem.complete_review(ticket, "SKIPPED")
        return {"saved": False, "status": "SKIPPED"}
    category = (category or "").upper()
    if category not in _LESSON_CATEGORIES:
        return {"saved": False, "error": f"category must be one of {', '.join(_LESSON_CATEGORIES)}"}
    lesson = (lesson or "").strip()
    if len(lesson) < 15:
        return {"saved": False, "error": "Lesson too short — be specific (15+ characters)."}
    score = max(0.0, min(1.0, float(usefulness_score)))
    ctx = rev.get("entry_context") or {}
    record = {
        "symbol": rev["symbol"], "direction": rev["direction"],
        "entry_price": rev.get("entry_price") or 0.0, "exit_price": rev.get("exit_price") or 0.0,
        "pnl_pct": rev.get("pnl_pct") or 0.0, "outcome": rev.get("outcome"),
        "fingerprint": fingerprint(ctx), "market_context": ctx,
        "agent_reasoning": f"MCP: {rev.get('reason_at_entry', '')}"[:2000],
        "lesson": lesson, "usefulness_score": score,
        "pattern_name": pattern_name or None, "category": category,
    }
    record.update(lesson_fields(ctx, float(rev.get("r_multiple") or 0.0), rev.get("entry_time")))
    saved = mem.save(record)
    mem.complete_review(ticket, "SAVED" if saved else "BELOW_THRESHOLD")
    return {"saved": bool(saved), "status": "SAVED" if saved else "BELOW_THRESHOLD",
            "usefulness_threshold": config.MEMORY_USEFULNESS_THRESHOLD}


def mt5b_equity() -> float:
    with _MT5_LOCK:
        _mt5_ready()
        return mt5b.get_account_equity()


# ── HTTP auth (ChatGPT supports OAuth or no auth — not API-key headers) ─────

class TokenAuth:
    """
    ASGI wrapper: every HTTP request must carry the secret token, either as
    'Authorization: Bearer <token>' (Claude/Codex) or '?token=<token>' in the
    URL (ChatGPT, which cannot send custom headers).
    """
    def __init__(self, app, token: str):
        self.app, self.token = app, token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"")
            given = auth[7:].strip() if auth[:7].lower() == b"bearer " else b""
            if not given:
                qs = parse_qs((scope.get("query_string") or b"").decode())
                given = (qs.get("token") or [""])[0].encode()
            if not given or not hmac.compare_digest(given, self.token):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body",
                            "body": json.dumps({"error": "missing or invalid token"}).encode()})
                return
        await self.app(scope, receive, send)


def build_http_app(token: str, public_hosts: list[str] | None = None):
    from mcp.server.transport_security import TransportSecuritySettings
    hosts = ["127.0.0.1:*", "localhost:*"] + [h for h in (public_hosts or [])]
    security = TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                         allowed_hosts=hosts, allowed_origins=["*"])
    app = server.streamable_http_app(transport_security=security)
    return TokenAuth(app, token)


def main(argv=None):
    ap = argparse.ArgumentParser(description="MCP server for the AI forex trading bot")
    ap.add_argument("--http", action="store_true", help="Serve over streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--public-host", action="append", default=[],
                    help="Host name of your HTTPS tunnel (e.g. abc.ngrok-free.app); repeatable")
    args = ap.parse_args(argv)

    # MCP clients start this server from their own working folder (Claude
    # Desktop has no setting for it). The bot's files (observability/,
    # memory DB, governor state) are relative paths, so work from the bot's
    # folder. FOREX_BOT_DIR overrides it (used by the tests).
    os.chdir(os.environ.get("FOREX_BOT_DIR") or os.path.dirname(os.path.abspath(__file__)))

    log.info(f"Trading mode: {_trading_mode()}  |  sizing: {getattr(config, 'SIZING_MODE', 'fixed')}")
    if args.http:
        token = os.environ.get("MCP_HTTP_TOKEN", "")
        if len(token) < 24:
            sys.exit("Set MCP_HTTP_TOKEN to a random secret of at least 24 characters "
                     "(e.g. py -c \"import secrets; print(secrets.token_urlsafe(32))\").")
        _STATE["transport"] = "http"
        import uvicorn
        log.info(f"Streamable HTTP on http://{args.host}:{args.port}/mcp  "
                 f"(trading over HTTP: {getattr(config, 'MCP_HTTP_ALLOW_TRADING', False)})")
        uvicorn.run(build_http_app(token, args.public_host), host=args.host, port=args.port,
                    log_level="warning")
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
