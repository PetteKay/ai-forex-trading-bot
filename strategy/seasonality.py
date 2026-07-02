"""
Seasonality — time-based confidence adjustment per symbol.

Returns a multiplier (0.80–1.20) and reason string applied to Fund Manager confidence.
Based on: London open surge, NY open momentum, Asian JPY tendencies, Friday position squaring.
"""

from datetime import datetime, timezone

_LONDON_OPEN = (8, 10)    # UTC hours
_NY_OPEN     = (13, 15)
_ASIAN_START = 22         # wraps midnight
_ASIAN_END   = 6
_FRIDAY_CLOSE_HOUR = 16

# Symbols that trade best during Asian session (22:00–06:00 UTC)
_ASIAN_SYMBOLS = {"USDJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY", "EURJPY", "GBPJPY", "JPN225", "AUDUSD", "NZDUSD"}
# Symbols that peak at London open (08:00–10:00 UTC)
_LONDON_SYMBOLS = {"XAUUSD", "XAGUSD", "GER40", "UK100", "GBPNZD", "GBPAUD", "GBPCAD", "GBPCHF", "EURNZD"}
# Symbols that peak at NY open (13:00–15:00 UTC)
_NY_SYMBOLS = {"NAS100", "US30", "USOIL", "NATGAS", "BTCUSD", "ETHUSD", "USDCAD"}


def _in_asian(hour: int) -> bool:
    return hour >= _ASIAN_START or hour < _ASIAN_END


def _in_session(hour: int, start: int, end: int) -> bool:
    return start <= hour < end


def get_adjustment(symbol: str, now: datetime | None = None) -> tuple[float, str]:
    """
    Returns (multiplier, reason).
    Multiplier range: 0.75–1.20. Neutral = 1.0.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    hour    = now.hour
    weekday = now.weekday()   # 0=Mon, 6=Sun
    sym     = symbol.upper().split(".")[0]  # strip broker suffix like .r / .pro

    factors: list[tuple[float, str]] = []

    # ── Day-of-week adjustments ───────────────────────────────────────────────
    if weekday == 6:  # Sunday
        factors.append((0.80, "Sunday: thin liquidity, wide spreads"))
    elif weekday == 0 and hour < 8:
        factors.append((0.85, "Monday pre-London: post-weekend gap risk"))
    elif weekday == 4 and hour >= _FRIDAY_CLOSE_HOUR:
        factors.append((0.85, "Friday close: position squaring, spreads widening"))

    # ── General time-of-day ───────────────────────────────────────────────────
    if _in_session(hour, *_LONDON_OPEN):
        factors.append((1.15, "London open: highest-volatility window, trend setups ideal"))
    elif _in_session(hour, *_NY_OPEN):
        factors.append((1.10, "NY open: second momentum surge, continuation setups valid"))
    elif hour in (12, 13) and weekday < 5:
        factors.append((0.90, "London/NY handover: directional chop, wait for NY commitment"))

    # ── Symbol-specific session alignment ─────────────────────────────────────
    if sym in _ASIAN_SYMBOLS:
        if _in_asian(hour):
            factors.append((1.10, f"{sym} in its strongest session (Asian 22:00–06:00 UTC)"))
        elif not _in_session(hour, *_LONDON_OPEN) and not _in_session(hour, *_NY_OPEN):
            factors.append((0.90, f"{sym} outside its preferred Asian session"))

    elif sym in _LONDON_SYMBOLS:
        if _in_session(hour, *_LONDON_OPEN):
            pass  # already captured by general London open boost above
        elif hour >= 16 or hour < 8:
            factors.append((0.90, f"{sym} trades best at London open — currently inactive"))

    elif sym in _NY_SYMBOLS:
        if _in_session(hour, *_NY_OPEN):
            pass  # already captured by general NY open boost
        elif not _in_session(hour, *_LONDON_OPEN):
            factors.append((0.92, f"{sym} prefers NY session — sub-optimal timing"))

    if not factors:
        return 1.0, "Standard trading hours"

    # Compound all factors
    composite = 1.0
    for mult, _ in factors:
        composite *= mult
    composite = round(max(0.75, min(1.25, composite)), 3)

    reasons = "; ".join(r for _, r in factors)
    return composite, reasons
