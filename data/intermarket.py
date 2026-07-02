"""
Inter-market confidence bias — maps macro signals to per-symbol directional adj.

Takes the already-fetched live_macro dict from data/macro_context.py
(no additional network calls) and returns a confidence adjustment for
a specific symbol + action based on DXY, Gold, Oil, VIX alignment.

Intermarket relationships used:
  DXY rising   → bearish EUR, GBP, AUD, NZD, Gold; bullish USD pairs
  DXY falling  → opposite
  Oil rising   → bullish CAD; bearish USDCAD
  Oil falling  → bearish CAD; bullish USDCAD
  Gold rising  → bullish XAUUSD; risk-off signal
  VIX HIGH+    → risk-off: fade AUD/NZD, buy CHF/JPY/Gold

Oil price is fetched here (CL=F via yfinance), cached 60 min.
"""

import logging
import time

log = logging.getLogger("InterMarket")

_oil_cache: dict   = {}
_oil_cache_ts: float = 0.0
_OIL_TTL = 3600  # refresh oil every 60 min


def _fetch_oil_trend() -> str:
    """
    Do the actual yfinance fetch for WTI crude trend. Runs inside an isolated
    child process (see data.process_isolation.run_isolated) — must stay a
    plain, module-level, picklable function. yfinance's curl_cffi backend has
    documented Windows access-violation crashes on network failures, which a
    Python try/except cannot catch; isolating this call means such a crash
    can only take down the child, never the live trading bot.
    """
    trend = "NEUTRAL"
    try:
        import yfinance as yf
        hist = yf.Ticker("CL=F").history(period="3d", interval="1h", auto_adjust=True)
        if not hist.empty:
            closes = hist["Close"].dropna()
            if len(closes) >= 2:
                pct = (float(closes.iloc[-1]) - float(closes.iloc[-2])) / float(closes.iloc[-2]) * 100
                trend = "RISING" if pct > 0.3 else "FALLING" if pct < -0.3 else "NEUTRAL"
    except Exception as e:
        log.debug(f"Oil fetch skipped: {e}")
    return trend


def _get_oil_trend() -> str:
    """Return 'RISING' | 'FALLING' | 'NEUTRAL' for WTI crude."""
    global _oil_cache, _oil_cache_ts
    now = time.time()
    if _oil_cache and (now - _oil_cache_ts) < _OIL_TTL:
        return _oil_cache.get("trend", "NEUTRAL")

    from data.process_isolation import run_isolated
    trend = run_isolated(_fetch_oil_trend, timeout=20.0, default="NEUTRAL")

    _oil_cache    = {"trend": trend}
    _oil_cache_ts = now
    return trend


# Symbols where DXY rise is a HEADWIND (these pairs fall when USD strengthens)
_DXY_HEADWIND = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "XAUUSD", "XAGUSD",
                 "GBPJPY", "EURJPY", "AUDJPY", "NZDJPY", "GBPNZD", "GBPAUD",
                 "GBPCAD", "GBPCHF", "EURNZD", "CADJPY"}

# Symbols where DXY rise is a TAILWIND (pair rises when USD strengthens)
_DXY_TAILWIND = {"USDCAD", "USDCHF", "USDJPY"}

# Risk-off assets (rise when VIX is HIGH/EXTREME)
_RISK_OFF_BULL = {"XAUUSD", "USDCHF"}

# Risk-on assets (fall when VIX is HIGH/EXTREME — should SELL these)
_RISK_OFF_BEAR = {"AUDUSD", "NZDUSD", "AUDJPY", "NZDJPY", "GBPJPY", "NAS100", "US30",
                  "BTCUSD", "ETHUSD", "USOIL", "GER40"}


def get_intermarket_bias(symbol: str, action: str, live_macro: dict) -> dict:
    """
    Compute intermarket confidence adjustment for a given symbol + action.

    Args:
        symbol:     e.g. "EURUSD"
        action:     "BUY" | "SELL"
        live_macro: dict from data.macro_context.get_macro_context()

    Returns:
        bias:           "BULLISH" | "BEARISH" | "NEUTRAL"
        confidence_adj: int  (positive = add, negative = subtract)
        reasons:        list[str]
    """
    if not live_macro:
        return {"bias": "NEUTRAL", "confidence_adj": 0, "reasons": []}

    sym = symbol.upper().split(".")[0]
    reasons = []
    score   = 0  # positive = bullish sym, negative = bearish sym

    # ── DXY ──────────────────────────────────────────────────────────────────
    dxy_trend = live_macro.get("dxy_trend", "")  # "STRENGTHENING" | "WEAKENING"
    if dxy_trend == "STRENGTHENING":
        if sym in _DXY_HEADWIND:
            score -= 1
            reasons.append(f"DXY strengthening → headwind {sym}")
        elif sym in _DXY_TAILWIND:
            score += 1
            reasons.append(f"DXY strengthening → tailwind {sym}")
    elif dxy_trend == "WEAKENING":
        if sym in _DXY_HEADWIND:
            score += 1
            reasons.append(f"DXY weakening → tailwind {sym}")
        elif sym in _DXY_TAILWIND:
            score -= 1
            reasons.append(f"DXY weakening → headwind {sym}")

    # ── Oil (fetched lazily from yfinance, cached 60 min) ────────────────────
    oil_trend = _get_oil_trend()
    if "CAD" in sym:
        if sym == "USDCAD":
            if oil_trend == "RISING":
                score -= 1
                reasons.append("Oil rising → CAD strength → USDCAD down")
            elif oil_trend == "FALLING":
                score += 1
                reasons.append("Oil falling → CAD weakness → USDCAD up")
        elif sym.endswith("CAD"):
            # e.g. GBPCAD, AUDCAD — CAD is the quote currency; CAD strength = pair falls
            if oil_trend == "RISING":
                score -= 1
                reasons.append(f"Oil rising → CAD strength → {sym} down (quote-CAD)")
            elif oil_trend == "FALLING":
                score += 1
                reasons.append(f"Oil falling → CAD weakness → {sym} up (quote-CAD)")
        else:
            # e.g. CADJPY — CAD is the base currency; CAD strength = pair rises
            if oil_trend == "RISING":
                score += 1
                reasons.append(f"Oil rising → CAD strength → {sym} up (base-CAD)")
            elif oil_trend == "FALLING":
                score -= 1
                reasons.append(f"Oil falling → CAD weakness → {sym} down (base-CAD)")
    if sym in ("USOIL", "UKOIL"):
        if oil_trend == "RISING":
            score += 1
            reasons.append("Oil momentum positive")
        elif oil_trend == "FALLING":
            score -= 1
            reasons.append("Oil momentum negative")

    # ── Gold ─────────────────────────────────────────────────────────────────
    gold_trend = live_macro.get("gold_trend", "")
    if sym == "XAUUSD":
        if gold_trend == "RISING":
            score += 1
            reasons.append("Gold momentum positive → bullish XAUUSD")
        elif gold_trend == "FALLING":
            score -= 1
            reasons.append("Gold momentum negative → bearish XAUUSD")

    # ── VIX ─────────────────────────────────────────────────────────────────
    vix_regime = live_macro.get("vix_regime", "")  # "CALM" | "NORMAL" | "ELEVATED" | "FEAR"
    if vix_regime in ("ELEVATED", "FEAR"):
        if sym in _RISK_OFF_BULL:
            score += 1
            reasons.append(f"VIX {vix_regime} → risk-off bid for {sym}")
        elif sym in _RISK_OFF_BEAR:
            score -= 1
            reasons.append(f"VIX {vix_regime} → risk-off headwind for {sym}")

    # ── Convert score to adj ─────────────────────────────────────────────────
    if score >= 2:
        bias, adj = "BULLISH", 4
    elif score == 1:
        bias, adj = "BULLISH", 2
    elif score <= -2:
        bias, adj = "BEARISH", -4
    elif score == -1:
        bias, adj = "BEARISH", -2
    else:
        bias, adj = "NEUTRAL", 0

    # Flip adj sign when action opposes the macro bias
    if action == "BUY"  and bias == "BEARISH":
        adj = -abs(adj)
    elif action == "SELL" and bias == "BULLISH":
        adj = -abs(adj)
    elif action == "BUY"  and bias == "BULLISH":
        adj =  abs(adj)
    elif action == "SELL" and bias == "BEARISH":
        adj =  abs(adj)

    return {"bias": bias, "confidence_adj": adj, "reasons": reasons}
