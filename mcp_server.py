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
import subprocess
import sys
import threading
import time
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
import bot_control                                   # noqa: E402
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
        "live mode. bot_status / start_bot / stop_bot / pause_trading / update_settings / "
        "retrain_ml_model / run_training control bot.py itself; only one of bot.py and this "
        "server trades the account at a time. Nothing here is financial advice."
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


def _acct():
    """The account lock (bot_control.py). Held by a running bot.py, or by this
    server while it trades / books (specs/BotControl.tla: lock)."""
    return bot_control.account_lock(bot_control.LOCK_PATH)


def _governor(equity: float) -> RiskGovernor:
    """Fresh governor from the shared state file, so restarts and the bot's limits
    apply. It writes the file only while this server holds the account lock — when
    bot.py runs, the bot owns that file (read-only here)."""
    bot_control.apply_overrides(config)
    return RiskGovernor(
        starting_equity    = equity,
        daily_loss_cap_pct = config.DAILY_LOSS_CAP_PCT,
        max_open_positions = config.MAX_OPEN_POSITIONS,
        max_orders_per_day = config.MAX_ORDERS_PER_DAY,
        max_drawdown_pct   = config.MAX_DRAWDOWN_PCT,
        allowed_sessions   = config.ALLOWED_SESSIONS,
        state_path         = getattr(config, "GOVERNOR_STATE_PATH", "observability/governor_state.json"),
        account_id         = mt5b.get_account_login(),
        persist            = _acct().held_here,
    )


def _book_closed_trades(gov: RiskGovernor) -> list[dict]:
    """
    Book trades that have closed since the last check — closed by close_position,
    by their stop loss / take profit, or while the server was not running — and
    learn from them exactly as bot.py does (trade_booking.py, shared with bot.py):

      daily loss cap (governor) · journal · setup blacklist · ML-gate training
      sample + online update · periodic retrain · Telegram alert · lesson

    Only while this server holds the account lock (call inside `_acct().hold()`):
    when bot.py runs it books every close itself, so nothing is booked twice
    (specs/BotControl.tla INV-BC-3).
    """
    if not _acct().held_here:
        return []
    from trade_booking import book_closed_trades
    try:
        tg = _telegram()
    except Exception:
        tg = None
    booked = book_closed_trades(journal=_journal(), memory=_memory(), governor=gov,
                                meta_gate=_meta_gate, mt5b=mt5b, magic=_magic(), cfg=config,
                                mt5_lock=_MT5_LOCK, telegram=tg, llm_state=_lazy)
    for b in booked:
        if b["ml_sample"]:
            _lazy["since_retrain"] = _lazy.get("since_retrain", 0) + 1
            if _lazy["since_retrain"] >= getattr(config, "META_RETRAIN_EVERY", 20):
                _meta_gate().train(_memory().get_all_for_training(),
                                   training_lookback_days=getattr(config, "ML_TRAINING_LOOKBACK_DAYS", 180))
                _lazy["since_retrain"] = 0
    return booked


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
    with _acct().hold() as mine:
        gov = _governor(equity)
        booked = _book_closed_trades(gov) if mine else []
        gstatus = gov.status
    return _jsonable({
        "equity": equity, "balance": balance,
        "newly_booked_closed_trades": booked,
        "open_positions": positions,
        "governor": gstatus,
        "bot": _bot_summary(),
        "trading_paused": bool(bot_control.read_control().get("paused")),
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

def _plan(symbol: str, direction: str, size_multiplier: float, mine: bool = None) -> dict:
    """mine: this server holds the account lock (call inside `_acct().hold()`)."""
    if mine is None:
        mine = _acct().held_here
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
    if mine:
        _book_closed_trades(gov)      # realised losses count before approving anything
    else:
        st = bot_control.bot_state()
        blockers.append(
            f"bot.py is running on this account (instance {st.get('instance') or '?'}) and does "
            f"the trading: MCP orders are off while it runs (one trader per account). Use "
            f"pause_trading / stop_bot, or close_position (carried out by the bot)."
            if st["running"] else "The account is busy (another MCP server is trading). Try again.")
    if bot_control.read_control().get("paused"):
        blockers.append("Trading is paused (pause_trading). Call resume_trading to allow orders.")
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
    with _acct().hold() as mine:
        plan = _plan(symbol, direction, size_multiplier, mine)
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
    with _TRADE_LOCK, _acct().hold() as mine:     # [McpAcquire ... McpRelease]
        return _place_locked(symbol, direction, size_multiplier, reason, confidence, mine)


def _place_locked(symbol: str, direction: str, size_multiplier: float, reason: str,
                  confidence: int = 0, mine: bool = None) -> dict:
    plan = _plan(symbol, direction, size_multiplier, mine)
    ctx, gov = plan["_ctx"], plan["_gov"]
    if not plan["allowed"]:
        return _jsonable({"placed": False, **plan})
    if _trading_mode() != "live":
        return _jsonable({"placed": False, "dry_run": True,
                          "note": "Dry run: config.MCP_TRADING_MODE is not 'live'. No order sent.",
                          **plan})
    if bot_control.read_control().get("paused"):            # [McpPlace: ~paused]
        return _jsonable({"placed": False, "error": "Trading was paused just now (pause_trading)."})
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
    bot's magic number). Dry run unless config.MCP_TRADING_MODE = "live". While
    bot.py runs, the bot carries out the close (one trader per account)."""
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
    st = bot_control.bot_state()
    if st["running"]:                                            # [RequestClose]
        inst = st["instance"]
        bot_control.request_close(inst, ticket)
        done = bot_control.wait_for(
            lambda s: s.get("instance") == inst and str(ticket) in (s.get("closes") or {}), 45)
        if done is None:
            return _jsonable({"closed": False, "requested": True, "position": pos,
                              "note": "Close sent to the running bot; it hasn't confirmed yet "
                                      "(it may be mid-tick). Check get_account_status shortly."})
        res = done["closes"][str(ticket)]
        return _jsonable({"closed": res == "closed", "by": f"bot.py {inst}", "result": res,
                          "position": pos, "note": "The bot books the result on its next tick."})
    with _acct().hold() as mine:                                  # [McpClose]
        if not mine:
            return {"closed": False, "error": "The account is busy (a bot is starting, or another "
                                              "MCP server is trading). Try again."}
        with _MT5_LOCK:
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
    equity = mt5b_equity()
    with _acct().hold() as mine:
        booked = _book_closed_trades(_governor(equity)) if mine else []
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


# ── Controlling bot.py (bot_control.py, specs/BotControl.tla) ────────────────

SAFE_CONTROL = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                               idempotent_hint=True, open_world_hint=False)
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_BOT_SCRIPT = os.path.join(_CODE_DIR, "bot.py")
_CHILDREN: dict[int, subprocess.Popen] = {}        # bot / trainer processes started here
_JOBS: dict[str, dict] = {}


def _reap() -> None:
    for pid, p in list(_CHILDREN.items()):
        if p.poll() is not None:
            _CHILDREN.pop(pid, None)


def _tail(path: str, lines: int) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            data = f.read().decode("utf-8", errors="replace")
        return data.splitlines()[-lines:]
    except OSError:
        return []


def _bot_summary() -> dict:
    _reap()
    st = bot_control.bot_state()
    s = st["status"]
    out = {"running": st["running"]}
    if st["running"]:
        out.update({k: s.get(k) for k in ("instance", "pid", "mode", "started", "heartbeat",
                                          "tick", "last_tick", "paused", "state")})
        try:
            hb = datetime_fromiso(s.get("heartbeat"))
            out["seconds_since_heartbeat"] = round(time.time() - hb.timestamp(), 1)
        except Exception:
            pass
    elif s.get("state"):
        out["last_run"] = {k: s.get(k) for k in ("instance", "mode", "started", "stopped", "state")}
    return out


def datetime_fromiso(v):
    from datetime import datetime
    return datetime.fromisoformat(str(v))


def _control_block_reason(loosening: bool) -> str | None:
    """Actions that can start trading are refused over HTTP unless the owner allowed
    it, and when MCP trading is off. Pausing / stopping is always allowed."""
    if not loosening:
        return None
    if _trading_mode() == "off":
        return "Trading tools are off (config.MCP_TRADING_MODE = 'off')."
    if _STATE["transport"] == "http" and not getattr(config, "MCP_HTTP_ALLOW_TRADING", False):
        return "Over HTTP this needs config.MCP_HTTP_ALLOW_TRADING = True."
    return None


def _spawn(args: list, log_path: str) -> subprocess.Popen:
    """Start a child that keeps running after this MCP server exits."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logf = open(log_path, "ab")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    kw = dict(cwd=os.getcwd(), stdout=logf, stderr=subprocess.STDOUT,
              stdin=subprocess.DEVNULL, env=env)
    try:
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            try:   # leave the MCP client's job object, so closing the app doesn't kill it
                p = subprocess.Popen(args, creationflags=flags | 0x01000000, **kw)
            except OSError:
                p = subprocess.Popen(args, creationflags=flags, **kw)
        else:
            p = subprocess.Popen(args, start_new_session=True, **kw)
    finally:
        logf.close()       # the child has its own handle
    _CHILDREN[p.pid] = p
    return p


@server.tool(annotations=READ_ONLY)
def bot_status() -> dict:
    """Is bot.py running? Its instance id, pid, demo/live mode, last heartbeat and
    tick, whether trading is paused, and the settings an MCP client has tightened."""
    fv = bot_control._file_values(config)
    ov = bot_control.read_overrides()
    return _jsonable({
        "bot": _bot_summary(),
        "trading_paused": bool(bot_control.read_control().get("paused")),
        "settings_overrides": {k: {"config_py": fv.get(k), "override": v,
                                   "effective": bot_control.effective(k, fv.get(k), v)}
                               for k, v in ov.items()},
        "mcp_trading_mode": _trading_mode(),
    })


@server.tool(annotations=READ_ONLY)
def get_bot_log(lines: int = 60, which: str = "bot") -> dict:
    """The last lines of bot.py's log. which="bot" (bot.log) or "console"
    (observability/bot_console.log: start-up errors, crashes)."""
    lines = max(1, min(int(lines), 400))
    path = "bot.log" if which != "console" else os.path.join("observability", "bot_console.log")
    return {"file": path, "lines": _tail(path, lines)}


@server.tool(annotations=DESTRUCTIVE)
def start_bot(live: bool = False) -> dict:
    """Start bot.py in the background (it keeps running if this app closes).
    live=False (default) runs it with --demo: it analyses and logs trades but sends
    no orders. live=True sends real orders and needs config.MCP_TRADING_MODE = "live"
    (set by you in config.py; it can't be changed from here). bot.py needs its own
    LLM key (config.LLM_API_KEY) for its agent desk."""
    blocked = _control_block_reason(True)
    if blocked:
        return {"started": False, "error": blocked}
    if live and _trading_mode() != "live":                      # [ACT-BC-7]
        return {"started": False, "error": "Live needs config.MCP_TRADING_MODE = 'live' in "
                                           "config.py (your decision; not settable from MCP)."}
    st = bot_control.bot_state()
    if st["running"]:
        return _jsonable({"started": False, "error": "bot.py is already running.",
                          "bot": _bot_summary()})
    args = [bot_control.python_exe(), _BOT_SCRIPT] + ([] if live else ["--demo"])
    console = os.path.join("observability", "bot_console.log")
    p = _spawn(args, console)
    ok = bot_control.wait_for(lambda s: s.get("pid") == p.pid and s.get("state") in
                              ("starting", "running"), 45, interval=0.2)
    if ok is None or p.poll() is not None:
        return {"started": False, "exit_code": p.poll(),
                "error": "bot.py did not start. Last console lines below.",
                "console": _tail(console, 25)}
    return _jsonable({"started": True, "mode": "live" if live else "demo", "pid": p.pid,
                      "instance": ok.get("instance"),
                      "note": "It now connects to MT5, loads memory and trains the ML gate; "
                              "bot_status shows when it is ticking."})


@server.tool(annotations=SAFE_CONTROL)
def stop_bot(force: bool = False, wait_seconds: int = 30) -> dict:
    """Stop the running bot.py. It finishes its current step and exits (open positions
    stay open with their stop loss / take profit; a restarted bot adopts them).
    force=True ends it immediately if it hasn't stopped within wait_seconds."""
    st = bot_control.bot_state()
    if not st["running"]:
        return {"stopped": False, "error": "No bot.py is running."}
    inst, pid = st["instance"], st["status"].get("pid")
    bot_control.request_stop(inst)                              # [RequestStop: names the instance]
    end = time.monotonic() + max(1, min(int(wait_seconds), 120))
    while time.monotonic() < end:
        _reap()
        if not bot_control.pid_alive(pid):
            return {"stopped": True, "how": "graceful", "instance": inst}
        time.sleep(0.5)
    if not force:
        return {"stopped": False, "stop_requested": True, "instance": inst,
                "note": "Still finishing its current step (an agent-desk round can take a "
                        "while). Call bot_status later, or stop_bot(force=True)."}
    cur = bot_control.read_status()
    if cur.get("instance") != inst or cur.get("pid") != pid:     # [ForceKill: same instance only]
        return {"stopped": True, "how": "graceful", "instance": inst}
    killed = bot_control.kill_pid(pid)
    _reap()
    return {"stopped": killed, "how": "forced", "instance": inst}


@server.tool(annotations=DESTRUCTIVE)
def restart_bot(live: bool = False, force: bool = False) -> dict:
    """Stop bot.py and start it again (e.g. to load new settings or a retrained model)."""
    blocked = _control_block_reason(True)
    if blocked:
        return {"restarted": False, "error": blocked}
    stopped = stop_bot(force=force)
    if bot_control.bot_state()["running"]:
        return {"restarted": False, "stop": stopped}
    started = start_bot(live=live)
    return {"restarted": bool(started.get("started")), "stop": stopped, "start": started}


def _wait_ack(version: int, timeout: float = 30) -> bool:
    st = bot_control.bot_state()
    if not st["running"]:
        return False
    inst = st["instance"]
    return bot_control.wait_for(lambda s: s.get("instance") == inst
                                and int(s.get("control_version", -1)) >= version, timeout) is not None


@server.tool(annotations=SAFE_CONTROL)
def pause_trading() -> dict:
    """Pause new trades — by bot.py and by MCP orders. Open positions keep being
    managed (trailing stop, exits). Doesn't touch the daily-loss kill switch."""
    c = bot_control.set_paused(True)                            # [SetPause(TRUE)]
    running = bot_control.bot_state()["running"]
    confirmed = _wait_ack(c["version"]) if running else None
    return {"paused": True, "bot_running": running, "bot_confirmed": confirmed,
            "note": ("The bot confirmed: it has no order in flight and won't send new ones."
                     if confirmed else "The bot will apply it at its next check (every 2 s while "
                     "idle, and before every order)." if running else "No bot running; MCP orders are paused.")}


@server.tool(annotations=DESTRUCTIVE)
def resume_trading() -> dict:
    """Allow new trades again after pause_trading. The risk governor still applies:
    this does NOT clear a daily-loss kill switch or drawdown halt (only the next day,
    or you, can)."""
    blocked = _control_block_reason(True)
    if blocked:
        return {"resumed": False, "error": blocked}
    c = bot_control.set_paused(False)
    running = bot_control.bot_state()["running"]
    return {"resumed": True, "bot_running": running,
            "bot_confirmed": _wait_ack(c["version"]) if running else None}


@server.tool(annotations=READ_ONLY)
def get_settings() -> dict:
    """The limits an MCP client may change, with config.py's value, any override and
    the value in use. Overrides can only make a limit stricter."""
    fv = bot_control._file_values(config)
    ov = bot_control.read_overrides()
    rules = {"max": "can only be lowered", "min": "can only be raised",
             "subset": "can only drop symbols"}
    return _jsonable({k: {"config_py": fv.get(k), "override": ov.get(k),
                          "in_use": bot_control.effective(k, fv.get(k), ov.get(k)),
                          "rule": rules[spec[0]]}
                      for k, spec in bot_control.TIGHTEN_ONLY.items()})


@server.tool(annotations=SAFE_CONTROL)
def update_settings(changes: dict) -> dict:
    """Tighten limits, e.g. {"RISK_PER_TRADE_PCT": 0.25, "MAX_OPEN_POSITIONS": 2,
    "SYMBOLS": ["EURUSD", "USDJPY"]}. Each value must be at least as strict as
    config.py (see get_settings). The MCP server uses them at once; bot.py on its
    next start (restart_bot)."""
    res = bot_control.set_overrides(config, changes)
    if res.get("saved"):                                        # [SetAiLimit]
        bot_control.apply_overrides(config)
        if bot_control.bot_state()["running"]:
            res["note"] = "Saved. bot.py is running: call restart_bot to apply them to it."
    return _jsonable(res)


@server.tool(annotations=DESTRUCTIVE)
def reset_settings(keys: list[str] | None = None) -> dict:
    """Remove overrides (all, or the keys given): back to config.py's values."""
    blocked = _control_block_reason(True)
    if blocked:
        return {"cleared": False, "error": blocked}
    res = bot_control.clear_overrides(keys)                     # [ClearAiLimit]
    bot_control.apply_overrides(config)
    return _jsonable(res)


@server.tool(annotations=SAFE_CONTROL)
def retrain_ml_model() -> dict:
    """Retrain the ML gate now from every closed trade and training sample in the
    bot's memory (it also retrains on its own every META_RETRAIN_EVERY trades).
    While bot.py runs, the bot retrains its own model."""
    st = bot_control.bot_state()
    if st["running"]:
        inst = st["instance"]
        _, seq = bot_control.request_retrain(inst)
        done = bot_control.wait_for(lambda s: s.get("instance") == inst
                                    and int(s.get("retrain_done", 0)) >= seq, 120)
        if done is None:
            return {"retrained": False, "requested": True,
                    "note": "Sent to the running bot; it hasn't finished yet (mid-tick?)."}
        return _jsonable({"retrained": True, "by": f"bot.py {inst}", **(done.get("retrain_result") or {})})
    _lazy.pop("meta_gate", None)
    g = _meta_gate()
    return {"retrained": bool(getattr(g, "_trained", False)), "by": "MCP server",
            "n_samples": getattr(g, "_n_samples", 0),
            "note": "" if getattr(g, "_trained", False) else
                    "Not enough training data yet: the gate uses its heuristic until it has enough."}


_TRAINERS = {
    "mt5_history":       "mt5_historical_trainer.py",
    "tick_csv":          "csv_historical_trainer.py",
    "trade_csv_lessons": "trainer.py",
}


@server.tool(annotations=SAFE_CONTROL)
def run_training(kind: str, symbol: str = "", bars: int = 5000, max_samples: int = 500,
                 path: str = "") -> dict:
    """Run one of the bot's trainers in the background; returns a job_id for
    get_training_job. kind:
      "mt5_history"  ML-gate samples from MT5 price history (needs MT5; bars, max_samples, symbol)
      "tick_csv"     ML-gate samples from MT5 tick-export CSVs in the folder `path`
      "trade_csv_lessons"  lessons from a CSV of your past trades at `path` (uses the LLM key)
    Then call retrain_ml_model so the gate uses the new samples."""
    import re
    import uuid
    if kind not in _TRAINERS:
        return {"started": False, "error": f"kind must be one of {list(_TRAINERS)}"}
    for j in _JOBS.values():
        if j["proc"].poll() is None:
            return {"started": False, "error": f"Training job {j['id']} is still running."}
    if symbol and not re.fullmatch(r"[A-Za-z0-9._#-]{3,20}", symbol):
        return {"started": False, "error": "symbol looks invalid"}
    bars, max_samples = max(100, min(int(bars), 100_000)), max(10, min(int(max_samples), 5000))
    args = [bot_control.python_exe(), os.path.join(_CODE_DIR, _TRAINERS[kind])]
    if kind == "mt5_history":
        args += ["--bars", str(bars), "--max-samples", str(max_samples)]
        if symbol:
            args += ["--symbol", symbol]
    elif kind == "tick_csv":
        if not path or not os.path.isdir(path):
            return {"started": False, "error": "path must be a folder with <SYMBOL>_TICK_*.csv files"}
        args += ["--dir", path, "--max-samples", str(max_samples)] + (["--symbol", symbol] if symbol else [])
    else:
        if not path or not os.path.isfile(path):
            return {"started": False, "error": "path must be a CSV file of past trades"}
        if not getattr(config, "LLM_API_KEY", ""):
            return {"started": False, "error": "trade_csv_lessons uses the LLM: set config.LLM_API_KEY"}
        args += ["--file", path]
    job_id = uuid.uuid4().hex[:8]
    log_path = os.path.join("observability", "jobs", f"{job_id}.log")
    p = _spawn(args, log_path)
    _JOBS[job_id] = {"id": job_id, "kind": kind, "proc": p, "log": log_path,
                     "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    return {"started": True, "job_id": job_id, "kind": kind}


@server.tool(annotations=READ_ONLY)
def get_training_job(job_id: str = "") -> dict:
    """Status and last output lines of a training job (the latest if no job_id)."""
    if not _JOBS:
        return {"error": "No training job has been started by this server."}
    j = _JOBS.get(job_id) or list(_JOBS.values())[-1]
    code = j["proc"].poll()
    out = {"job_id": j["id"], "kind": j["kind"], "started": j["started"],
           "running": code is None, "exit_code": code, "output": _tail(j["log"], 30)}
    if code == 0:
        out["next"] = "Call retrain_ml_model so the ML gate uses the new samples."
    return out


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
