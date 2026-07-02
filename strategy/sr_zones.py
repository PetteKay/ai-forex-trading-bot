"""
Support/Resistance Zone Memory — Multi-Touch Level Detection.

Identifies price zones that have been tested 3+ times from the same side
without a decisive breakout. These zones represent institutional order clusters.

Algorithm:
  1. Detect swing highs (resistance pivots) and swing lows (support pivots)
     from the most recent LOOKBACK H1 bars.
  2. Cluster nearby pivots within a tolerance band (zone_width per symbol).
  3. A cluster with >= MIN_TOUCHES pivot tests is a confirmed S/R zone.
  4. Return zones sorted by strength (touch count + recency).

Usage:
    from strategy.sr_zones import find_zones, check_sr_zone
    zones = find_zones("EURUSD", h1_df)
    result = check_sr_zone("EURUSD", current_price, zones)
"""

import logging
from typing import NamedTuple

import pandas as pd

log = logging.getLogger("SRZones")

_MIN_TOUCHES = 3
_LOOKBACK    = 200   # H1 bars to analyse
_PIVOT_N     = 5     # bars each side must confirm the pivot

_ZONE_WIDTH: dict[str, float] = {
    "EURUSD": 0.0015, "GBPUSD": 0.0020, "AUDUSD": 0.0012,
    "NZDUSD": 0.0012, "USDCAD": 0.0015, "USDCHF": 0.0015,
    "USDJPY": 0.15,   "GBPJPY": 0.25,   "EURJPY": 0.20,
    "AUDJPY": 0.18,   "NZDJPY": 0.15,   "CADJPY": 0.15,
    "CHFJPY": 0.15,   "GBPNZD": 0.0030, "GBPAUD": 0.0025,
    "GBPCAD": 0.0025, "GBPCHF": 0.0025, "EURNZD": 0.0025,
    "XAUUSD": 3.0,    "XAGUSD": 0.15,
    "USOIL":  0.30,   "NATGAS": 0.03,
    "NAS100": 30.0,   "US30":   30.0,   "GER40":  15.0,
    "JPN225": 50.0,   "BTCUSD": 300.0,  "ETHUSD": 15.0,
}
_DEFAULT_WIDTH = 0.0015

_PIP: dict[str, float] = {
    "USDJPY": 0.01, "GBPJPY": 0.01, "EURJPY": 0.01, "AUDJPY": 0.01,
    "NZDJPY": 0.01, "CADJPY": 0.01, "CHFJPY": 0.01,
    "XAUUSD": 0.01, "XAGUSD": 0.01, "USOIL": 0.01, "NATGAS": 0.01,
    "NAS100": 1.0,  "US30":   1.0,  "GER40":  1.0, "JPN225": 1.0,
    "BTCUSD": 1.0,  "ETHUSD": 1.0,
}


class SRZone(NamedTuple):
    price:     float
    touches:   int
    zone_type: str    # "SUPPORT" | "RESISTANCE"
    strength:  float  # 0-1


def _pivots_high(highs: list[float], n: int) -> list[tuple[int, float]]:
    out = []
    for i in range(n, len(highs) - n):
        if all(highs[i] >= highs[i - j] for j in range(1, n + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, n + 1)):
            out.append((i, highs[i]))
    return out


def _pivots_low(lows: list[float], n: int) -> list[tuple[int, float]]:
    out = []
    for i in range(n, len(lows) - n):
        if all(lows[i] <= lows[i - j] for j in range(1, n + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, n + 1)):
            out.append((i, lows[i]))
    return out


def _cluster(
    pivots: list[tuple[int, float]],
    tol: float,
    zone_type: str,
    n_bars: int,
) -> list[SRZone]:
    if not pivots:
        return []
    sorted_p = sorted(pivots, key=lambda p: p[1])
    zones: list[SRZone] = []
    used = [False] * len(sorted_p)
    for i, (idx_i, price_i) in enumerate(sorted_p):
        if used[i]:
            continue
        cluster = [(idx_i, price_i)]
        used[i] = True
        for j in range(i + 1, len(sorted_p)):
            if not used[j] and abs(sorted_p[j][1] - price_i) <= tol:
                cluster.append(sorted_p[j])
                used[j] = True
        if len(cluster) < _MIN_TOUCHES:
            continue
        zone_price  = sum(p[1] for p in cluster) / len(cluster)
        recency     = max(p[0] for p in cluster) / max(n_bars, 1)
        touches     = len(cluster)
        strength    = min(1.0, round(touches / 6 * 0.7 + recency * 0.3, 3))
        zones.append(SRZone(round(zone_price, 8), touches, zone_type, strength))
    return zones


def find_zones(symbol: str, df: pd.DataFrame) -> list[SRZone]:
    """Find confirmed multi-touch S/R zones. Returns list sorted by strength (strongest first)."""
    sym  = symbol.upper().split(".")[0]
    tol  = _ZONE_WIDTH.get(sym, _DEFAULT_WIDTH)
    try:
        bars = df.tail(_LOOKBACK).copy()
        bars.reset_index(drop=True, inplace=True)
        highs = bars["high"].tolist()
        lows  = bars["low"].tolist()
        n     = len(bars)
    except Exception as e:
        log.debug(f"SR zones data error: {e}")
        return []

    if n < _PIVOT_N * 2 + 1:
        return []

    ph = _pivots_high(highs, _PIVOT_N)
    pl = _pivots_low(lows,   _PIVOT_N)

    resistance = _cluster(ph, tol, "RESISTANCE", n)
    support    = _cluster(pl, tol, "SUPPORT",    n)

    all_zones = sorted(resistance + support, key=lambda z: z.strength, reverse=True)
    return all_zones[:10]


def check_sr_zone(
    symbol: str,
    price: float,
    zones: list[SRZone],
    proximity_mult: float = 2.0,
) -> dict:
    """
    Check whether current price is near any confirmed S/R zone.

    Returns:
        at_sr_zone:    bool
        zone_type:     str  — "SUPPORT" | "RESISTANCE" | ""
        zone_price:    float
        zone_strength: float (0-1)
        touches:       int
        distance_pips: float
    """
    sym = symbol.upper().split(".")[0]
    tol = _ZONE_WIDTH.get(sym, _DEFAULT_WIDTH) * proximity_mult
    pip = _PIP.get(sym, 0.0001)

    empty = {
        "at_sr_zone": False, "zone_type": "", "zone_price": 0.0,
        "zone_strength": 0.0, "touches": 0, "distance_pips": 0.0,
    }
    if not zones:
        return empty

    nearest  = min(zones, key=lambda z: abs(z.price - price))
    dist     = abs(nearest.price - price)
    dist_pip = round(dist / pip, 1) if pip > 0 else 0.0

    if dist > tol:
        return {**empty, "zone_price": nearest.price, "zone_strength": nearest.strength,
                "touches": nearest.touches, "distance_pips": dist_pip}

    return {
        "at_sr_zone":    True,
        "zone_type":     nearest.zone_type,
        "zone_price":    nearest.price,
        "zone_strength": nearest.strength,
        "touches":       nearest.touches,
        "distance_pips": dist_pip,
    }
