"""
Key Level Detection — identifies significant price zones.

Detects:
  1. Round numbers (institutional order clustering at 50-pip intervals)
  2. Previous week high / low (bank reference levels)
  3. Fibonacci retracement levels (38.2%, 50%, 61.8% from recent swing high/low)

Being AT a key level AND aligned with trade direction is a high-quality entry filter.
Being AT a key level AGAINST the trade direction is a warning (resistance when buying).

Usage:
  from strategy.key_levels import check
  kl = check("EURUSD", 1.08742, h1_df)
"""

import math
from datetime import datetime, timezone, timedelta

import pandas as pd

# ── Fibonacci constants ───────────────────────────────────────────────────────
_FIB_RATIOS     = [0.236, 0.382, 0.500, 0.618, 0.786]
_FIB_KEY        = {0.382, 0.500, 0.618}   # major Fib levels (stronger signal)
_FIB_SWING_BARS = 50                       # bars used to identify recent swing
_FIB_TOLERANCE  = 0.25                     # fraction of Fib-gap to count as "at" level

# ── Round number intervals (in price units, not pips) ────────────────────────
# Calibrated to significant institutional order zones per asset class.
_ROUND_INTERVALS: dict[str, float] = {
    "EURUSD": 0.0050, "GBPUSD": 0.0050, "AUDUSD": 0.0050,
    "NZDUSD": 0.0050, "USDCAD": 0.0050, "USDCHF": 0.0050,
    "USDJPY": 0.50,   "GBPJPY": 0.50,   "EURJPY": 0.50,
    "AUDJPY": 0.50,   "NZDJPY": 0.50,   "CADJPY": 0.50,
    "CHFJPY": 0.50,   "GBPNZD": 0.0050, "GBPAUD": 0.0050,
    "GBPCAD": 0.0050, "GBPCHF": 0.0050, "EURNZD": 0.0050,
    "XAUUSD": 10.0,   "XAGUSD": 0.50,
    "USOIL":  1.0,    "NATGAS": 0.10,
    "NAS100": 100.0,  "US30":   100.0,  "GER40":  50.0,
    "JPN225": 100.0,  "BTCUSD": 1000.0, "ETHUSD": 50.0,
}
_DEFAULT_INTERVAL = 0.0050

# ── Proximity tolerance: fraction of interval to count as "at" the level ─────
_NEAR_FRACTION  = 0.15   # within 15% of interval = AT the level
_WEEKLY_FRACTION = 0.50  # weekly high/low: within 50% of interval


def _get_interval(sym: str) -> float:
    return _ROUND_INTERVALS.get(sym, _DEFAULT_INTERVAL)


def _pip_size(sym: str) -> float:
    if "JPY" in sym or sym.startswith("XAU") or sym.startswith("XAG"):
        return 0.01
    if sym in ("USOIL", "NATGAS"):
        return 0.01
    if sym in ("NAS100", "US30", "GER40", "JPN225"):
        return 1.0
    if sym in ("BTCUSD", "ETHUSD"):
        return 1.0
    return 0.0001


def _nearest_round_level(price: float, interval: float) -> tuple[float, float]:
    """Returns (nearest_level, distance_in_price_units)."""
    lower = math.floor(price / interval) * interval
    upper = lower + interval
    d_lower = abs(price - lower)
    d_upper = abs(price - upper)
    if d_lower <= d_upper:
        return round(lower, 8), d_lower
    return round(upper, 8), d_upper


def _prev_week_high_low(df: pd.DataFrame) -> tuple[float, float]:
    """
    Extract previous week's high and low from an OHLC DataFrame.
    df must have a 'time' column or DatetimeIndex.
    Returns (high, low) or (0, 0) if insufficient data.
    """
    try:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df.index = pd.to_datetime(df["time"], utc=True)
            else:
                return 0.0, 0.0
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        now = datetime.now(timezone.utc)
        this_monday = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        last_monday = this_monday - timedelta(weeks=1)

        prev = df[(df.index >= last_monday) & (df.index < this_monday)]
        if len(prev) < 5:
            return 0.0, 0.0
        return float(prev["high"].max()), float(prev["low"].min())
    except Exception:
        return 0.0, 0.0


def _swing_high_low(df: pd.DataFrame, n_bars: int = _FIB_SWING_BARS) -> tuple[float, float, int, int]:
    """
    Find recent swing high and low from last n_bars of OHLC data.
    Returns (swing_high, swing_low, high_idx, low_idx).
    high_idx/low_idx: position of the swing within the slice (0 = oldest bar).
    """
    try:
        bars = df.tail(n_bars).copy()
        bars.reset_index(drop=True, inplace=True)
        highs = bars["high"].tolist()
        lows  = bars["low"].tolist()
        if not highs:
            return 0.0, 0.0, 0, 0
        high_idx = int(max(range(len(highs)), key=lambda i: highs[i]))
        low_idx  = int(min(range(len(lows)),  key=lambda i: lows[i]))
        return highs[high_idx], lows[low_idx], high_idx, low_idx
    except Exception:
        return 0.0, 0.0, 0, 0


def _fib_levels(high: float, low: float) -> list[float]:
    """Compute Fibonacci retracement levels between swing high and swing low."""
    move = high - low
    if move <= 0:
        return []
    return [round(high - ratio * move, 8) for ratio in _FIB_RATIOS]


def check_fibonacci(symbol: str, price: float, df: pd.DataFrame) -> dict:
    """
    Check whether price is at or near a Fibonacci retracement level.

    Returns:
        at_fib_level:       bool
        fib_ratio:          float | None  — e.g. 0.618
        fib_label:          str           — e.g. "61.8%"
        fib_price:          float
        fib_distance_pips:  float
        fib_proximity_score: float        — 0-1 (1 = right at a major Fib level)
        fib_is_major:       bool          — True for 38.2%, 50%, 61.8%
        swing_high:         float
        swing_low:          float
    """
    sym      = symbol.upper().split(".")[0]
    pip      = _pip_size(sym)
    interval = _get_interval(sym)

    empty = {
        "at_fib_level": False, "fib_ratio": None, "fib_label": "",
        "fib_price": 0.0, "fib_distance_pips": 0.0,
        "fib_proximity_score": 0.0, "fib_is_major": False,
        "swing_high": 0.0, "swing_low": 0.0,
    }

    swing_high, swing_low, _, _ = _swing_high_low(df)
    if swing_high <= 0 or swing_low >= swing_high:
        return empty

    levels = _fib_levels(swing_high, swing_low)
    if not levels:
        return empty

    # Gap between adjacent Fib levels (approximate: use 38.2%→50% gap as reference)
    move    = swing_high - swing_low
    fib_gap = move * (0.500 - 0.382)   # ~11.8% of the move
    tol     = fib_gap * _FIB_TOLERANCE

    best_dist  = float("inf")
    best_idx   = -1
    for i, lvl in enumerate(levels):
        d = abs(price - lvl)
        if d < best_dist:
            best_dist, best_idx = d, i

    dist_pips = round(best_dist / pip, 1) if pip > 0 else 0.0

    if best_dist > tol:
        # Not AT a Fib level, but report proximity score for ML feature
        prox = max(0.0, 1.0 - best_dist / (fib_gap * 2))
        return {
            **empty,
            "fib_price":           round(levels[best_idx], 8),
            "fib_distance_pips":   dist_pips,
            "fib_proximity_score": round(prox, 3),
            "swing_high":          swing_high,
            "swing_low":           swing_low,
        }

    ratio    = _FIB_RATIOS[best_idx]
    label    = f"{ratio * 100:.1f}%"
    is_major = ratio in _FIB_KEY
    prox     = round(1.0 - best_dist / tol, 3) if tol > 0 else 1.0

    return {
        "at_fib_level":        True,
        "fib_ratio":           ratio,
        "fib_label":           label,
        "fib_price":           round(levels[best_idx], 8),
        "fib_distance_pips":   dist_pips,
        "fib_proximity_score": min(1.0, prox),
        "fib_is_major":        is_major,
        "swing_high":          swing_high,
        "swing_low":           swing_low,
    }


def check(symbol: str, price: float, df: pd.DataFrame) -> dict:
    """
    Check whether price is at or near a key level.

    Args:
        symbol: e.g. "EURUSD" or "EURUSD.r"
        price:  current mid-price
        df:     H1 (or HTF) OHLC DataFrame used to compute prev-week levels

    Returns dict:
        at_key_level    bool   — True when price is within the tolerance band
        level_type      str    — "ROUND_NUMBER" | "PREV_WEEK_HIGH" | "PREV_WEEK_LOW" | ""
        level_price     float  — the exact key level price
        distance_pips   float  — distance to the nearest key level in pips
        signal_alignment str   — "SUPPORT" (near level, price approached from above)
                                  "RESISTANCE" (near level, price approached from below)
                                  "NONE" (not at a key level)

    Interpretation for trade entry:
        BUY + SUPPORT     = ideal entry (buying at support level)
        SELL + RESISTANCE = ideal entry (selling at resistance level)
        BUY + RESISTANCE  = counter-signal (buying into resistance)
        SELL + SUPPORT    = counter-signal (selling into support)

    Return dict also includes fibonacci fields from check_fibonacci().
    """
    sym      = symbol.upper().split(".")[0]
    interval = _get_interval(sym)
    pip      = _pip_size(sym)
    tol_near = interval * _NEAR_FRACTION
    tol_wkly = interval * _WEEKLY_FRACTION

    candidates: list[dict] = []

    # 1. Nearest round number
    round_lvl, round_dist = _nearest_round_level(price, interval)
    if round_dist <= tol_near:
        candidates.append({"type": "ROUND_NUMBER", "price": round_lvl, "dist": round_dist})

    # 2. Previous week high / low
    wk_high, wk_low = _prev_week_high_low(df)
    if wk_high > 0 and abs(price - wk_high) <= tol_wkly:
        candidates.append({"type": "PREV_WEEK_HIGH", "price": wk_high, "dist": abs(price - wk_high)})
    if wk_low > 0 and abs(price - wk_low) <= tol_wkly:
        candidates.append({"type": "PREV_WEEK_LOW", "price": wk_low, "dist": abs(price - wk_low)})

    # Distance to nearest round level (always reported even when not AT it)
    nearest_pips = round(round_dist / pip, 1) if pip > 0 else 0.0

    # 3. Fibonacci retracement levels (always computed, merged into return dict)
    fib = check_fibonacci(symbol, price, df)

    if not candidates:
        return {
            "at_key_level":        False,
            "level_type":          "",
            "level_price":         round_lvl,
            "distance_pips":       nearest_pips,
            "signal_alignment":    "NONE",
            **fib,
        }

    best = min(candidates, key=lambda c: c["dist"])
    dist_pips = round(best["dist"] / pip, 1) if pip > 0 else 0.0

    # SUPPORT = price is at or below the level (level acts as floor)
    # RESISTANCE = price is at or above the level (level acts as ceiling)
    alignment = "SUPPORT" if price <= best["price"] else "RESISTANCE"

    return {
        "at_key_level":        True,
        "level_type":          best["type"],
        "level_price":         round(best["price"], 8),
        "distance_pips":       dist_pips,
        "signal_alignment":    alignment,
        **fib,
    }
