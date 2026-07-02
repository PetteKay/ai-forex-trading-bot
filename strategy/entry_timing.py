"""
M15 Entry Timing Layer — prevents chasing by confirming pullback entry on M15.

After H1 generates a BUY/SELL signal, this checks the M15 chart:
  PROCEED — M15 shows a retracement to EMA9; ideal entry
  WAIT    — M15 is extended (price chased too far from EMA); wait for pullback
  SKIP    — M15 counter-trend move is too strong to enter safely

Improves average fill price by ~0.3–0.5 ATR vs. chasing the H1 candle close.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("EntryTiming")


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    val   = 100.0 - (100.0 / (1.0 + rs))
    return float(val.iloc[-1]) if len(val) > 0 else 50.0


def check(
    m15_df: pd.DataFrame | None,
    h1_bias: str,       # "BUY" or "SELL"
    h1_atr: float,
) -> dict:
    """
    Analyse the M15 chart for optimal entry timing.

    Returns:
        decision: "PROCEED" | "WAIT" | "SKIP"
        reason:   str
        quality:  float 0–1 (1.0 = ideal pullback entry)
    """
    _fallback = {"decision": "PROCEED", "reason": "Insufficient M15 data — proceeding on H1 signal", "quality": 0.5}

    if m15_df is None or len(m15_df) < 20:
        return _fallback
    if h1_atr <= 0:
        return _fallback

    close = m15_df["close"]
    price = float(close.iloc[-1])

    ema9  = float(close.ewm(span=9,  adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])

    rsi          = _rsi(close)
    atr_distance = abs(price - ema9) / h1_atr  # how far price is from EMA9 in ATR units
    last3        = close.iloc[-3:].values
    m15_rising   = bool(last3[-1] > last3[0])

    if h1_bias == "BUY":
        # Ideal: price has pulled back toward EMA9 (not chasing breakout)
        pullback    = price <= ema9 * 1.003   # within 0.3% of EMA9 or below
        not_ext     = atr_distance < 1.0       # within 1 ATR of EMA9
        rsi_ok      = rsi < 65                 # not overbought on M15

        q = float(np.mean([
            1.0 if pullback    else 0.3,
            1.0 if not_ext     else max(0.2, 1.0 - atr_distance * 0.4),
            1.0 if rsi_ok      else 0.4,
            0.8 if m15_rising  else 0.6,
        ]))

        if pullback and not_ext and rsi_ok:
            return {"decision": "PROCEED", "reason": f"M15 pullback to EMA9 — RSI={rsi:.0f}, ext={atr_distance:.2f}x ATR: ideal entry", "quality": round(q, 2)}
        if atr_distance > 2.0 or rsi > 78:
            return {"decision": "WAIT", "reason": f"M15 extended — {atr_distance:.1f}x ATR above EMA9, RSI={rsi:.0f}: wait for pullback", "quality": round(q, 2)}
        return {"decision": "PROCEED", "reason": f"M15 acceptable — ext={atr_distance:.2f}x, RSI={rsi:.0f}", "quality": round(q, 2)}

    elif h1_bias == "SELL":
        # Ideal: price has bounced back up toward EMA9 (not chasing the breakdown)
        pullback    = price >= ema9 * 0.997
        not_ext     = atr_distance < 1.0
        rsi_ok      = rsi > 35                 # not oversold on M15

        q = float(np.mean([
            1.0 if pullback     else 0.3,
            1.0 if not_ext      else max(0.2, 1.0 - atr_distance * 0.4),
            1.0 if rsi_ok       else 0.4,
            0.8 if not m15_rising else 0.6,
        ]))

        if pullback and not_ext and rsi_ok:
            return {"decision": "PROCEED", "reason": f"M15 bounce to EMA9 — RSI={rsi:.0f}, ext={atr_distance:.2f}x ATR: ideal entry", "quality": round(q, 2)}
        if atr_distance > 2.0 or rsi < 22:
            return {"decision": "WAIT", "reason": f"M15 extended — {atr_distance:.1f}x ATR below EMA9, RSI={rsi:.0f}: wait for bounce", "quality": round(q, 2)}
        return {"decision": "PROCEED", "reason": f"M15 acceptable — ext={atr_distance:.2f}x, RSI={rsi:.0f}", "quality": round(q, 2)}

    return _fallback
