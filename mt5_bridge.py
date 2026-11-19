"""
MetaTrader 5 Bridge — all interaction with MT5 goes through here.

Requires: MetaTrader 5 terminal installed and logged in to your broker account.
Package:  pip install MetaTrader5
"""

import MetaTrader5 as mt5
import pandas as pd
import logging
import time
from datetime import datetime, timedelta, timezone

log = logging.getLogger("MT5Bridge")

# MT5 timeframe string → constant
TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

# Verified from MQL5 docs: SYMBOL_FILLING bitmask values
_FILLING_FOK    = 1
_FILLING_IOC    = 2
_FILLING_RETURN = 4

# Complete retcode map for human-readable logging
RETCODE_NAMES = {
    10004: "REQUOTE",          10006: "REJECT",
    10007: "CANCEL",           10008: "PLACED",
    10009: "DONE",             10010: "DONE_PARTIAL",
    10011: "ERROR",            10012: "TIMEOUT",
    10013: "INVALID",          10014: "INVALID_VOLUME",
    10015: "INVALID_PRICE",    10016: "INVALID_STOPS",
    10017: "TRADE_DISABLED",   10018: "MARKET_CLOSED",
    10019: "NO_MONEY",         10020: "PRICE_CHANGED",
    10021: "PRICE_OFF",        10022: "INVALID_EXPIRATION",
    10024: "TOO_MANY_REQUESTS",10028: "LOCKED",
    10029: "FROZEN",           10030: "INVALID_FILL",
    10031: "CONNECTION",       10032: "ONLY_REAL",
    10033: "LIMIT_ORDERS",     10034: "LIMIT_VOLUME",
    10035: "INVALID_ORDER",    10036: "POSITION_CLOSED",
    10040: "LIMIT_POSITIONS",  10042: "LONG_ONLY",
    10043: "SHORT_ONLY",       10044: "CLOSE_ONLY",
    10046: "HEDGE_PROHIBITED",
}

# Retcodes where the order reached the broker and is considered placed
_RETCODE_SUCCESS = {mt5.TRADE_RETCODE_DONE, 10010}  # DONE + DONE_PARTIAL
# Retcodes where a price-retry is appropriate
_RETCODE_RETRY   = {10004, 10020, 10021}             # REQUOTE, PRICE_CHANGED, PRICE_OFF


def connect() -> bool:
    """Start MT5 and verify connection. Returns True on success."""
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    info    = mt5.terminal_info()
    account = mt5.account_info()
    if info is None or account is None:
        # initialize() can succeed while the terminal is still logging in —
        # treat missing terminal/account info as a failed connection.
        log.error(f"MT5 initialized but terminal/account info unavailable: {mt5.last_error()}")
        mt5.shutdown()
        return False
    log.info(f"Connected to MT5: {info.name} | Account: {account.login} | Broker: {account.company}")
    log.info(f"Balance: {account.balance:.2f} {account.currency} | Equity: {account.equity:.2f}")
    return True


def reconnect(max_attempts: int = 5) -> bool:
    """Reconnect to MT5 with exponential backoff. Call after detecting connection loss."""
    for attempt in range(max_attempts):
        mt5.shutdown()
        delay = min(2 ** attempt, 60)
        log.info(f"MT5 reconnect attempt {attempt+1}/{max_attempts} in {delay}s...")
        time.sleep(delay)
        if connect():
            log.info("MT5 reconnected successfully.")
            return True
    log.error("MT5 reconnect failed after all attempts.")
    return False


def disconnect():
    mt5.shutdown()
    log.info("MT5 disconnected.")


def resolve_symbol(base_symbol: str) -> str:
    """
    Resolve exact broker symbol name. Handles broker suffixes (.r, .pro, .a, etc.).
    Returns the exact symbol name as the broker lists it, or base_symbol if not found.
    """
    symbols = mt5.symbols_get()
    if not symbols:
        return base_symbol
    names = [s.name for s in symbols]
    if base_symbol in names:
        return base_symbol
    # Prefix match: find first symbol that starts with base_symbol
    for name in names:
        if name.startswith(base_symbol):
            log.warning(f"Symbol '{base_symbol}' not found; resolved to '{name}' (broker suffix).")
            return name
    log.warning(f"Symbol '{base_symbol}' not found in broker symbols list.")
    return base_symbol


def _detect_filling_mode(symbol: str) -> int:
    """
    Auto-detect the order filling mode supported by the broker for this symbol.
    Returns the mt5.ORDER_FILLING_* constant to use.
    Verified from MQL5 docs: symbol_info().filling_mode is a bitmask
    (1=FOK, 2=IOC, 4=RETURN).
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC  # safest fallback

    filling = info.filling_mode
    if filling & _FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    elif filling & _FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    elif filling & _FILLING_RETURN:
        return mt5.ORDER_FILLING_RETURN
    return mt5.ORDER_FILLING_IOC  # last resort


def get_account_equity() -> float:
    info = mt5.account_info()
    return float(info.equity) if info else 0.0


def get_account_balance() -> float:
    """Realized balance only — excludes floating P&L of open positions."""
    info = mt5.account_info()
    return float(info.balance) if info else 0.0


def get_pip_value(symbol: str, volume: float) -> float:
    """
    USD value of 1 pip for a given lot size. Handles standard pairs, JPY crosses,
    and XAU/Oil. Used for R-multiple outcome classification.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return volume * 10.0  # fallback: ~$1/pip per 0.1 lot
    tick_val  = info.trade_tick_value   # USD value of 1 tick per 1 lot
    tick_size = info.trade_tick_size    # e.g., 0.00001 for EURUSD
    sym_upper = symbol.upper()
    if "JPY" in sym_upper:
        pip_size = 0.01
    elif sym_upper.startswith(("XAU", "GOLD", "XAG")):
        pip_size = 0.01   # XAG added — was falling through to 0.0001 (100x off)
    else:
        pip_size = 0.0001
    pip_value_per_lot = tick_val * (pip_size / tick_size)
    return pip_value_per_lot * volume


def get_risk_usd(symbol: str, entry_price: float, sl_price: float, volume: float) -> float:
    """
    Exact USD risk from entry to stop-loss using MT5's own profit calculator.
    Falls back to pip-value approximation if calc is unavailable.
    Returns a positive number (absolute risk amount).
    """
    if sl_price <= 0 or entry_price <= 0:
        return 0.0
    # MT5's profit calculator is broker-accurate (handles swap, point-value quirks)
    profit = mt5.order_calc_profit(
        mt5.ORDER_TYPE_BUY,
        symbol,
        volume,
        entry_price,
        sl_price,
    )
    if profit is not None:
        return abs(float(profit))
    # Fallback: pip-value approximation
    pv = get_pip_value(symbol, volume)
    sym_upper = symbol.upper()
    pip_size = 0.01 if ("JPY" in sym_upper or sym_upper.startswith("XAU")) else 0.0001
    return (abs(entry_price - sl_price) / pip_size) * pv


def get_candles(symbol: str, timeframe: str, count: int = 200,
                include_forming: bool = False) -> pd.DataFrame:
    """
    Fetch the last `count` closed candles for a symbol.
    Returns a DataFrame with columns: time, open, high, low, close, volume

    MT5 numbers bars from present to past and bar 0 is the current, still-forming
    bar (MQL5 docs: copy_rates_from_pos / CopyRates). Starting at 1 returns only
    closed bars, so signals do not change while the candle is still forming.
    Pass include_forming=True only when the live bar is wanted on purpose.
    """
    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    start_pos = 0 if include_forming else 1
    rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No candle data for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )


def get_open_positions(symbol: str, magic: int) -> list[dict]:
    """Return all open positions for this symbol placed by this bot."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return []
    return [
        {
            "ticket":  p.ticket,
            "type":    "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume":  p.volume,
            "price":   p.price_open,
            "sl":      p.sl,
            "tp":      p.tp,
            "profit":  p.profit,
        }
        for p in positions if p.magic == magic
    ]


def get_all_open_positions(magic: int) -> list[dict]:
    """Return all open positions for this bot across every symbol."""
    positions = mt5.positions_get()
    if not positions:
        return []
    return [
        {
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type":   "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume": p.volume,
            "price":  p.price_open,
            "sl":     p.sl,
            "tp":     p.tp,
            "profit": p.profit,
            "time":   int(getattr(p, "time", 0) or 0),   # open time, terminal timebase
        }
        for p in positions if p.magic == magic
    ]


def terminal_time_to_utc(ts: int, symbol: str, now_utc=None):
    """
    Convert a position/deal timestamp from the terminal's timebase to UTC.

    MQL5 forum answers disagree on whether these timestamps are UTC or broker
    server time, so the offset is MEASURED instead of assumed: the symbol's
    latest tick time minus the real UTC clock, rounded to the nearest 30 min
    (≈0 if the terminal already uses UTC). If the tick is stale (market closed),
    the gap is not close to a 30-min multiple and None is returned.
    """
    from datetime import datetime as _dt, timezone as _tz
    if not ts:
        return None
    now_utc = now_utc or _dt.now(_tz.utc)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or not getattr(tick, "time", 0):
        return None
    raw     = float(tick.time) - now_utc.timestamp()
    offset  = round(raw / 1800.0) * 1800.0
    if abs(raw - offset) > 300:          # tick older than 5 min → can't measure
        return None
    dt = _dt.fromtimestamp(ts - offset, tz=_tz.utc)
    return min(dt, now_utc)


def get_unrealised_pnl(symbol: str, magic: int) -> float:
    """Total unrealised P&L across all open positions for this bot."""
    positions = get_open_positions(symbol, magic)
    return sum(p["profit"] for p in positions)


def get_closed_trade(ticket: int, max_wait_secs: float = 5.0) -> dict | None:
    """
    Query MT5 deal history for a completed trade by position ticket.
    Returns real entry_price, exit_price, pnl, swap, commission, net_pnl.
    None if history not found within max_wait_secs.

    Non-blocking design: tries immediately, then waits at most max_wait_secs
    total before returning None so the main loop is not stalled.
    Background reconciliation can retry later for the 3-hour indexing delay cases.
    """
    # Retry schedule: 0s then one short retry — total wait stays under max_wait_secs
    retry_waits = [0, min(max_wait_secs, 4.0)]

    for wait in retry_waits:
        if wait > 0:
            log.debug(f"MT5 deal history not yet indexed for ticket {ticket} — waiting {wait:.0f}s...")
            time.sleep(wait)

        # Primary: direct position-ID lookup (fastest when indexed)
        deals = mt5.history_deals_get(position=ticket)

        if not deals:
            # Fallback: date-range scan (handles cases where position-ID index lags)
            now     = datetime.now(timezone.utc)
            from_dt = now - timedelta(days=30)
            all_deals = mt5.history_deals_get(
                date_from=from_dt,
                date_to=now + timedelta(hours=1),
            )
            deals = [d for d in (all_deals or []) if d.position_id == ticket]

        if deals:
            break

    if not deals:
        log.warning(
            f"MT5 deal history unavailable for ticket {ticket} after retries. "
            "This can take up to 3 hours to appear — P&L will be estimated."
        )
        return None

    deals      = sorted(deals, key=lambda d: d.time)
    entry_deal = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_IN),  None)
    exit_deal  = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT), None)

    if not entry_deal or not exit_deal:
        log.warning(f"Incomplete deal set for ticket {ticket} (missing IN or OUT deal).")
        return None

    total_pnl  = sum(d.profit     for d in deals)
    total_swap = sum(d.swap       for d in deals)
    total_comm = sum(d.commission for d in deals)

    return {
        "ticket":      ticket,
        "entry_price": entry_deal.price,
        "exit_price":  exit_deal.price,
        "volume":      exit_deal.volume,
        "pnl":         total_pnl,
        "swap":        total_swap,
        "commission":  total_comm,
        "net_pnl":     total_pnl + total_swap + total_comm,
        "entry_time":  datetime.utcfromtimestamp(entry_deal.time),
        "exit_time":   datetime.utcfromtimestamp(exit_deal.time),
    }


def place_order(
    symbol: str,
    direction: str,   # "BUY" or "SELL"
    lot_size: float,
    sl_pips: float,
    tp_pips: float,
    magic: int,
    comment: str = "AI-Bot",
    max_retries: int = 3,
) -> dict | None:
    """
    Place a market order with stop loss and take profit.
    Returns the order result dict on success, None on failure.

    Fixes:
      - Auto-detects filling mode from symbol_info().filling_mode bitmask
        (VERIFIED BUG: hardcoded IOC causes retcode=10030 on many brokers)
      - Retries on REQUOTE (10004) and PRICE_CHANGED (10020) with fresh price
      - Falls back to FOK then RETURN if IOC rejected (10030)
      - Logs all retcodes with human-readable names
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        log.error(f"Symbol {symbol} not found.")
        return None

    if not info.visible:
        mt5.symbol_select(symbol, True)

    point        = info.point
    filling_mode = _detect_filling_mode(symbol)

    for attempt in range(max_retries):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            log.error(f"No tick data for {symbol}.")
            return None

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price      = tick.ask
            sl         = round(price - sl_pips * point, info.digits)
            tp         = round(price + tp_pips * point, info.digits)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price      = tick.bid
            sl         = round(price + sl_pips * point, info.digits)
            tp         = round(price - tp_pips * point, info.digits)

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(lot_size),
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "magic":        magic,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        retcode = result.retcode if result else None
        retname = RETCODE_NAMES.get(retcode, str(retcode))

        if result and retcode in _RETCODE_SUCCESS:
            log.info(
                f"Order placed: {direction} {lot_size} {symbol} @ {price} "
                f"| SL={sl} TP={tp} | Ticket={result.order} | Fill={retname}"
            )
            return {"ticket": result.order, "price": price, "sl": sl, "tp": tp}

        if retcode in _RETCODE_RETRY:
            log.warning(f"Order requote/price change ({retname}), retrying...")
            time.sleep(1)
            continue

        if retcode == 10030:  # INVALID_FILL — try a different filling mode
            if filling_mode == mt5.ORDER_FILLING_IOC:
                filling_mode = mt5.ORDER_FILLING_FOK
                log.warning("IOC filling rejected (10030) — trying FOK.")
            elif filling_mode == mt5.ORDER_FILLING_FOK:
                filling_mode = mt5.ORDER_FILLING_RETURN
                log.warning("FOK filling rejected (10030) — trying RETURN.")
            else:
                log.error(f"All filling modes rejected for {symbol}. Order failed.")
                return None
            continue

        log.error(
            f"Order failed: retcode={retcode} ({retname}) | "
            f"symbol={symbol} dir={direction} lot={lot_size} | {mt5.last_error()}"
        )
        return None

    log.error(f"Order failed after {max_retries} attempts ({symbol} {direction}).")
    return None


def close_position(
    ticket: int,
    symbol: str,
    volume: float,
    position_type: str,
    magic: int = 0,
    max_retries: int = 3,
) -> bool:
    """
    Close an open position by ticket.
    Fixes: auto-detect filling mode, retry on requote, handle all retcodes.
    """
    filling_mode = _detect_filling_mode(symbol)

    for attempt in range(max_retries):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        if position_type == "BUY":
            close_type = mt5.ORDER_TYPE_SELL
            price      = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price      = tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "magic":        magic,
            "comment":      "AIBot-close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result  = mt5.order_send(request)
        retcode = result.retcode if result else None
        retname = RETCODE_NAMES.get(retcode, str(retcode))

        if result and retcode in _RETCODE_SUCCESS:
            log.info(f"Position {ticket} closed ({retname}).")
            return True

        if retcode in _RETCODE_RETRY:
            log.warning(f"Close requote ({retname}) for ticket {ticket}, retrying...")
            time.sleep(1)
            continue

        if retcode == 10030:
            if filling_mode == mt5.ORDER_FILLING_IOC:
                filling_mode = mt5.ORDER_FILLING_FOK
            elif filling_mode == mt5.ORDER_FILLING_FOK:
                filling_mode = mt5.ORDER_FILLING_RETURN
            else:
                break
            continue

        log.error(f"Close failed for ticket {ticket}: retcode={retcode} ({retname})")
        return False

    log.error(f"Close failed for ticket {ticket} after {max_retries} attempts.")
    return False


def get_candles_mtf(symbol: str, timeframes: list[str], count: int = 200) -> dict[str, pd.DataFrame]:
    """Fetch candles for multiple timeframes. Returns {tf_string: DataFrame}."""
    result = {}
    for tf in timeframes:
        try:
            result[tf] = get_candles(symbol, tf, count)
        except Exception as e:
            log.warning(f"Could not fetch {symbol} {tf}: {e}")
    return result


def modify_position_sl(ticket: int, symbol: str, new_sl: float) -> bool:
    """Modify the stop loss of an existing open position."""
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return False
    p       = pos[0]
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   symbol,
        "position": ticket,
        "sl":       new_sl,
        "tp":       p.tp,
    }
    result  = mt5.order_send(request)
    retcode = result.retcode if result else None
    if result and retcode in _RETCODE_SUCCESS:
        log.info(f"SL moved to {new_sl:.5f} for ticket {ticket}")
        return True
    retname = RETCODE_NAMES.get(retcode, str(retcode))
    log.warning(f"SL modify failed for ticket {ticket}: {retcode} ({retname})")
    return False


def is_connected() -> bool:
    """Quick liveness check — returns False if MT5 terminal is not responding."""
    try:
        info = mt5.terminal_info()
        return info is not None and info.connected
    except Exception:
        return False


def get_live_price(symbol: str) -> float | None:
    """Current bid (the price MT5 bars are built from), or None if no tick."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or not getattr(tick, "bid", 0):
        return None
    return float(tick.bid)


def get_spread_pips(symbol: str) -> float:
    """
    Return the current bid-ask spread in pips.
    Uses broker live tick data. Returns 0.0 if data unavailable.
    Pip sizes: JPY pairs = 0.01; XAU/Gold = 0.01; everything else = 0.0001.
    """
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        return 0.0
    spread_points = tick.ask - tick.bid
    if spread_points <= 0:
        return 0.0
    sym_upper = symbol.upper()
    if "JPY" in sym_upper:
        pip_size = 0.01
    elif sym_upper.startswith("XAU") or sym_upper.startswith("GOLD"):
        pip_size = 0.01
    elif sym_upper.startswith("XAG"):
        pip_size = 0.01
    else:
        pip_size = 0.0001
    return round(spread_points / pip_size, 1)
