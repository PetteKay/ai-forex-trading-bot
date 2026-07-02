"""
Classic Pivot Point Levels — weekly and monthly.

Formula (floor pivots):
  PP = (H + L + C) / 3
  R1 = 2*PP - L    S1 = 2*PP - H
  R2 = PP + (H-L)  S2 = PP - (H-L)
  R3 = PP + 2*(H-L) S3 = PP - 2*(H-L)

Watched by millions of participants → act as magnetic S/R.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import NamedTuple


class PivotSet(NamedTuple):
    pp:  float
    r1:  float
    r2:  float
    r3:  float
    s1:  float
    s2:  float
    s3:  float
    source: str   # "WEEKLY" | "MONTHLY"


def _compute(high: float, low: float, close: float, source: str) -> PivotSet:
    pp = (high + low + close) / 3
    rng = high - low
    return PivotSet(
        pp=pp,
        r1=round(2 * pp - low,  8),
        r2=round(pp + rng,      8),
        r3=round(pp + 2 * rng,  8),
        s1=round(2 * pp - high, 8),
        s2=round(pp - rng,      8),
        s3=round(pp - 2 * rng,  8),
        source=source,
    )


def _prev_hlc(df: pd.DataFrame, period: str) -> tuple[float, float, float]:
    try:
        d = df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            if "time" in d.columns:
                d.index = pd.to_datetime(d["time"], utc=True)
            else:
                return 0.0, 0.0, 0.0
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")

        now = datetime.now(timezone.utc)
        if period == "WEEKLY":
            this_mon = (now - timedelta(days=now.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            start = this_mon - timedelta(weeks=1)
            end   = this_mon
        else:
            this_first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if this_first.month == 1:
                start = this_first.replace(year=this_first.year - 1, month=12)
            else:
                start = this_first.replace(month=this_first.month - 1)
            end = this_first

        sub = d[(d.index >= start) & (d.index < end)]
        if len(sub) < 5:
            return 0.0, 0.0, 0.0
        return float(sub["high"].max()), float(sub["low"].min()), float(sub["close"].iloc[-1])
    except Exception:
        return 0.0, 0.0, 0.0


def get_pivots(df: pd.DataFrame) -> list[PivotSet]:
    """Compute weekly and monthly pivot sets from H1 OHLC data."""
    result = []
    for period in ("WEEKLY", "MONTHLY"):
        h, l, c = _prev_hlc(df, period)
        if h > 0 and l < h:
            result.append(_compute(h, l, c, period))
    return result


_PIP: dict[str, float] = {
    "USDJPY": 0.01, "GBPJPY": 0.01, "EURJPY": 0.01, "AUDJPY": 0.01,
    "NZDJPY": 0.01, "CADJPY": 0.01, "CHFJPY": 0.01,
    "XAUUSD": 0.01, "XAGUSD": 0.01,
    "NAS100": 1.0, "US30": 1.0, "GER40": 1.0, "JPN225": 1.0,
    "BTCUSD": 1.0, "ETHUSD": 1.0, "USOIL": 0.01, "NATGAS": 0.001,
}


def check_pivot_levels(symbol: str, price: float, df: pd.DataFrame,
                        proximity_pct: float = 0.002) -> dict:
    """
    Check if price is within proximity_pct of any weekly/monthly pivot level.

    Returns:
        at_pivot:      bool
        pivot_type:    str  — "PP" | "R1".."R3" | "S1".."S3"
        pivot_price:   float
        pivot_source:  str  — "WEEKLY" | "MONTHLY"
        pivot_signal:  str  — "SUPPORT" | "RESISTANCE" | "NONE"
        distance_pips: float
    """
    sym = symbol.upper().split(".")[0]
    pip = _PIP.get(sym, 0.0001)
    tol = price * proximity_pct

    empty = {"at_pivot": False, "pivot_type": "", "pivot_price": 0.0,
             "pivot_source": "", "pivot_signal": "NONE", "distance_pips": 0.0}

    pivots = get_pivots(df)
    if not pivots:
        return empty

    best_dist  = float("inf")
    best_level = 0.0
    best_type  = ""
    best_source = ""

    for ps in pivots:
        for label, lvl in [("PP", ps.pp), ("R1", ps.r1), ("R2", ps.r2), ("R3", ps.r3),
                            ("S1", ps.s1), ("S2", ps.s2), ("S3", ps.s3)]:
            if lvl <= 0:
                continue
            d = abs(price - lvl)
            if d < best_dist:
                best_dist, best_level, best_type, best_source = d, lvl, label, ps.source

    dist_pips = round(best_dist / pip, 1) if pip else 0.0

    if best_dist > tol:
        return {**empty, "pivot_price": best_level, "distance_pips": dist_pips}

    # Support levels: S1/S2/S3/PP  |  Resistance: R1/R2/R3
    signal = "RESISTANCE" if best_type.startswith("R") else "SUPPORT"

    return {
        "at_pivot":      True,
        "pivot_type":    best_type,
        "pivot_price":   round(best_level, 8),
        "pivot_source":  best_source,
        "pivot_signal":  signal,
        "distance_pips": dist_pips,
    }
