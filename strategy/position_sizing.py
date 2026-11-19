"""
Position sizing.

Two modes (config.SIZING_MODE):

  "fixed"     lot = LOT_SIZE × size_multiplier, rounded to 2 dp, minimum 0.01.
              This is the original behaviour. At the default LOT_SIZE = 0.01,
              every multiplier below 1.5 still rounds to 0.01 (verified), so the
              Kelly / HMM / volatility / loss-streak adjustments cannot change
              the lot. They only take effect with a larger LOT_SIZE.

  "risk_pct"  lot = (equity × RISK_PER_TRADE_PCT% × size_multiplier)
                    ÷ (loss per 1 lot if the stop is hit)
              rounded DOWN to the broker's volume step and capped at
              MAX_LOT_SIZE and the broker maximum. If the result is below the
              broker minimum, the trade is skipped rather than rounded up, so
              the size adjustments are never overridden into more risk.

The loss per lot uses MT5's tick value and tick size:
    loss_per_lot = sl_distance_in_price / tick_size × tick_value
"""

import math


def _floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    # small epsilon guards against float error (e.g. 0.03 / 0.01 = 2.9999999)
    return math.floor(x / step + 1e-9) * step


def compute_lot(
    mode: str,
    size_mult: float,
    *,
    base_lot: float = 0.01,
    equity: float = 0.0,
    risk_pct: float = 0.5,
    sl_distance: float = 0.0,
    tick_value: float = 0.0,
    tick_size: float = 0.0,
    vol_min: float = 0.01,
    vol_step: float = 0.01,
    vol_max: float = 100.0,
    max_lot: float | None = None,
) -> tuple[float, str]:
    """
    Return (lot, note). lot == 0.0 means "do not trade" and note says why.
    sl_distance is the stop distance in PRICE units (e.g. 0.0020 for 20 pips EURUSD).
    """
    size_mult = max(0.0, float(size_mult))
    if size_mult <= 0:
        return 0.0, "size multiplier is 0"

    if mode == "fixed":
        lot = max(0.01, round(base_lot * size_mult, 2))
        return lot, f"fixed: {base_lot} × {size_mult:.2f} → {lot}"

    if mode != "risk_pct":
        raise ValueError(f"Unknown SIZING_MODE: {mode!r} (use 'fixed' or 'risk_pct')")

    if equity <= 0 or sl_distance <= 0 or tick_value <= 0 or tick_size <= 0:
        return 0.0, ("risk_pct: missing data "
                     f"(equity={equity}, sl={sl_distance}, tick_value={tick_value}, tick_size={tick_size})")

    risk_usd     = equity * (risk_pct / 100.0) * size_mult
    loss_per_lot = sl_distance / tick_size * tick_value
    raw_lot      = risk_usd / loss_per_lot
    lot          = _floor_to_step(raw_lot, vol_step)
    caps         = [vol_max] + ([max_lot] if max_lot else [])
    lot          = min(lot, *caps)
    lot          = round(lot, 8)
    if lot < vol_min - 1e-12:
        return 0.0, (f"risk_pct: {raw_lot:.4f} lots for ${risk_usd:.2f} risk is below the "
                     f"broker minimum {vol_min} — skipped (rounding up would add risk)")
    return lot, (f"risk_pct: ${risk_usd:.2f} risk ({risk_pct}% × {size_mult:.2f}) / "
                 f"${loss_per_lot:.2f} per lot → {lot}")


def kelly_fraction(p_win: float, payoff_b: float, fraction: float) -> float:
    """Fractional Kelly stake. 0.0 means no edge at these odds."""
    if payoff_b <= 0:
        return 0.0
    k = (p_win * payoff_b - (1.0 - p_win)) / payoff_b
    return max(0.0, k) * fraction
