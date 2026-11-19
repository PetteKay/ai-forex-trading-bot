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
    )


def _get(name, factory):
    if name not in _lazy:
        _lazy[name] = factory()
    return _lazy[name]


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
    return _jsonable({
        "equity": equity, "balance": balance,
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
        "regime": ctx["regime"].get("regime"), "_ctx": ctx, "_gov": gov,
    }


@server.tool(annotations=READ_ONLY)
def preview_trade(symbol: str, direction: str, size_multiplier: float = 1.0) -> dict:
    """Check a BUY or SELL against every rule the bot uses (risk governor, calendar
    blackout, duplicate and correlation guards, spread, position sizing) and show
    the lot, stop loss, take profit and money at risk. Sends no order."""
    plan = _plan(symbol, direction, size_multiplier)
    return _jsonable(plan)


@server.tool(annotations=DESTRUCTIVE)
def place_trade(symbol: str, direction: str, size_multiplier: float = 1.0,
                reason: str = "") -> dict:
    """Open a market BUY or SELL with the bot's ATR stop loss and take profit.
    Runs every check in preview_trade first and refuses if any fails. In
    config.MCP_TRADING_MODE = "dry_run" (the default) it only reports what it
    would do. `reason`: a short note saved with the trade."""
    blocked = _trading_block_reason()
    if blocked:
        return {"placed": False, "error": blocked}
    with _TRADE_LOCK:
        return _place_locked(symbol, direction, size_multiplier, reason)


def _place_locked(symbol: str, direction: str, size_multiplier: float, reason: str) -> dict:
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
    try:
        _journal().log_trade_open(
            ticket=result["ticket"], symbol=plan["symbol"], direction=plan["direction"],
            entry_price=result["price"], sl=result["sl"], tp=result["tp"],
            volume=plan["lot"], confidence=0, regime=plan["regime"] or "",
            fingerprint=market_fingerprint(ctx), size_multiplier=size_multiplier)
    except Exception as e:
        log.warning(f"Journal write failed: {e}")
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
    return _jsonable({"closed": bool(ok), "position": pos})


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
