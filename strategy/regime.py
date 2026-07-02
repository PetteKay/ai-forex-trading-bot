"""
Regime Classifier — detects the current market regime before any trade.

Four regimes:
  TRENDING_UP    — directional uptrend with momentum; EMA strategies work
  TRENDING_DOWN  — directional downtrend with momentum
  RANGING        — oscillating between support/resistance; mean-reversion works
  VOLATILE       — high-volatility, indeterminate; reduce size or skip

Methods used:
  ADX (Average Directional Index)  — trend strength (> 25 = trending)
  Hurst Exponent (R/S analysis)    — > 0.55 persistent trend, < 0.45 mean-reverting
  ATR percentile                   — normalized volatility regime
  EMA alignment                    — directional confirmation
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("Regime")

try:
    from ml.fast_hurst import compute_hurst_fast as _fast_hurst
    _USE_FAST_HURST = True
except Exception:
    _USE_FAST_HURST = False


# ── ADX ─────────────────────────────────────────────────────────────────────
def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    # Compute raw directional movement
    up_move   = high.diff()
    down_move = -low.diff()

    # Use numpy where instead of in-place boolean indexing — avoids
    # pandas SettingWithCopyWarning and silent failure in newer pandas.
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move,  0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=high.index)
    minus_dm_s = pd.Series(minus_dm, index=high.index)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm_s.ewm(span=period,  adjust=False).mean() / atr
    minus_di = 100 * minus_dm_s.ewm(span=period, adjust=False).mean() / atr
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val  = dx.ewm(span=period, adjust=False).mean()
    return adx_val, plus_di, minus_di


# Minimum bars required for Hurst to be statistically reliable.
# VERIFIED: R/S Hurst on <500 returns gives noisy, unreliable estimates.
# With 250 H1 bars the bot had ~50 returns — far below this threshold.
_HURST_MIN_BARS = 200


# ── Hurst Exponent (R/S method) ─────────────────────────────────────────────
def _hurst(series: np.ndarray, min_lag: int = 2, max_lag: int = 50) -> tuple[float, bool]:
    """
    Compute Hurst exponent using R/S analysis.
    Returns (hurst_value, reliable) where reliable=False means insufficient data.
    H > 0.55 → persistent trend (momentum)
    H ~ 0.5  → random walk
    H < 0.45 → mean-reverting

    VERIFIED: requires at least ~200 return observations for reliable H estimate.
    With <200 bars, falls back returning (0.5, False) — caller should rely on ADX only.
    """
    n = len(series)
    if n < _HURST_MIN_BARS:
        log.debug(f"Hurst skipped: only {n} bars, need {_HURST_MIN_BARS}. Using ADX-only regime.")
        return 0.5, False

    # Numba-JIT fast path — bypasses the python-level R/S loop entirely
    if _USE_FAST_HURST:
        try:
            h = _fast_hurst(series, min_lag, max_lag)
            return float(np.clip(h, 0.0, 1.0)), True
        except Exception:
            pass  # fall through to numpy

    lags      = range(min_lag, min(max_lag, n // 2))
    rs_values = []
    for lag in lags:
        chunks = [series[i:i + lag] for i in range(0, n - lag, lag)]
        if not chunks:
            continue
        rs_list = []
        for chunk in chunks:
            mean       = np.mean(chunk)
            deviations = np.cumsum(chunk - mean)
            r          = deviations.max() - deviations.min()
            s          = np.std(chunk, ddof=1) if np.std(chunk, ddof=1) > 0 else 1e-10
            rs_list.append(r / s)
        rs_values.append(np.mean(rs_list))

    if len(rs_values) < 3:
        return 0.5, False

    log_lags = np.log(list(lags[:len(rs_values)]))
    log_rs   = np.log(rs_values)
    try:
        hurst, _ = np.polyfit(log_lags, log_rs, 1)
        return float(np.clip(hurst, 0.0, 1.0)), True
    except Exception:
        return 0.5, False


# ── Kaufman Efficiency Ratio ─────────────────────────────────────────────────
def _efficiency_ratio(close: np.ndarray, n: int = 10) -> float:
    """
    Kaufman Efficiency Ratio (ER) — cheap trend-efficiency signal.
    ER = |net price change over N bars| / sum(|individual bar changes|)
    1.0 = perfectly directional; 0.0 = pure chop.
    Complements ADX by reacting faster (no smoothing lag).
    """
    if len(close) < n + 1:
        return 0.5
    net_change  = abs(float(close[-1]) - float(close[-n - 1]))
    sum_changes = float(np.sum(np.abs(np.diff(close[-n - 1:]))))
    return float(net_change / sum_changes) if sum_changes > 1e-10 else 0.5


# ── Main classifier ──────────────────────────────────────────────────────────
def classify(df: pd.DataFrame, atr_percentile: float, adx_period: int = 14) -> dict:
    """
    Classify the current market regime.

    Args:
        df: OHLCV DataFrame (needs at least 100 bars for reliable Hurst)
        atr_percentile: current ATR percentile (0-100) from indicators.compute()
        adx_period: period for ADX calculation

    Returns dict with:
        regime: TRENDING_UP | TRENDING_DOWN | RANGING | VOLATILE
        adx: float
        hurst: float
        atr_percentile: float
        confidence: float (0-1)
        description: str
    """
    if len(df) < 50:
        return {
            "regime": "UNKNOWN", "adx": 0, "hurst": 0.5,
            "atr_percentile": atr_percentile, "confidence": 0,
            "description": "Insufficient data for regime classification",
        }

    adx_series, plus_di, minus_di = _adx(df, adx_period)
    adx_val   = float(adx_series.iloc[-1])
    plus_val  = float(plus_di.iloc[-1])
    minus_val = float(minus_di.iloc[-1])

    # Use all available bars for Hurst; _hurst() signals if it's insufficient
    close_arr              = df["close"].values
    returns                = np.diff(np.log(close_arr + 1e-10))
    hurst_val, hurst_valid = _hurst(returns)

    # Kaufman Efficiency Ratio — fast trend-efficiency complement to ADX
    er_val = _efficiency_ratio(close_arr)

    # EMA alignment (fast vs trend)
    ema9   = df["close"].ewm(span=9,   adjust=False).mean().iloc[-1]
    ema21  = df["close"].ewm(span=21,  adjust=False).mean().iloc[-1]
    ema200 = df["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    price  = df["close"].iloc[-1]

    bull_aligned = (price > ema9 > ema21 > ema200)
    bear_aligned = (price < ema9 < ema21 < ema200)

    # ── Regime decision logic ─────────────────────────────────────────────
    # Primary threshold: ADX > 25 = trending (industry standard).
    # Hurst enhancement: if Hurst is very strong (> 0.64), persistent trend
    # signal is reliable even with lower ADX (>= 15). This catches slow-building
    # trends where price persistence leads ADX.
    ADX_TREND_THRESHOLD      = 25
    ADX_HURST_BOOST_THRESHOLD = 10   # ADX floor when Hurst confirms strong persistence
    ADX_WEAK_THRESHOLD       = 20

    volatile = atr_percentile > 75 and adx_val > 35

    # When Hurst is unreliable (insufficient data), gate on ADX alone
    if hurst_valid:
        # Standard trending: ADX > 25 AND moderate Hurst
        normal_trending = adx_val > ADX_TREND_THRESHOLD and hurst_val > 0.52
        # Hurst-boosted: strong persistence (> 0.64) with some ADX (>= 10)
        # — catches slow persistent trends that ADX hasn't confirmed yet
        hurst_boosted   = hurst_val > 0.64 and adx_val >= ADX_HURST_BOOST_THRESHOLD
        # ER-boosted: Efficiency Ratio > 0.55 is an independent directional signal
        # (reacts faster than ADX; use with ADX >= 10 to filter random noise)
        er_boosted      = er_val > 0.55 and adx_val >= ADX_HURST_BOOST_THRESHOLD
        trending = normal_trending or hurst_boosted or er_boosted
    else:
        # ADX-only fallback — also allow ER to confirm at low ADX
        trending = adx_val > ADX_TREND_THRESHOLD or (er_val > 0.60 and adx_val >= ADX_HURST_BOOST_THRESHOLD)

    if volatile:
        regime     = "VOLATILE"
        confidence = min(1.0, (atr_percentile - 75) / 25 + (adx_val - 35) / 30)
        desc       = f"High-volatility (ADX={adx_val:.1f}, ATR%={atr_percentile:.0f})"
    elif trending and (bull_aligned or plus_val > minus_val):
        regime      = "TRENDING_UP"
        hurst_boost = hurst_valid and hurst_val > 0.64 and adx_val < ADX_TREND_THRESHOLD
        if hurst_boost:
            # Hurst-led trend: base confidence from Hurst strength, capped at moderate level
            base_conf = (hurst_val - 0.64) * 2
            adx_conf  = max(0, (adx_val - ADX_HURST_BOOST_THRESHOLD) / (ADX_TREND_THRESHOLD - ADX_HURST_BOOST_THRESHOLD))
            confidence = min(0.65, base_conf + adx_conf * 0.3)
            desc = (f"Trending up via Hurst persistence (ADX={adx_val:.1f} weak, "
                    f"Hurst={hurst_val:.2f} strong, +DI>{plus_val:.1f}) [Hurst-led]")
        else:
            base_conf  = (adx_val - ADX_TREND_THRESHOLD) / 25
            hurst_conf = (hurst_val - 0.5) * 2 if hurst_valid else 0
            confidence = min(1.0, base_conf + hurst_conf)
            desc       = (f"Trending up (ADX={adx_val:.1f}, "
                          f"Hurst={'N/A' if not hurst_valid else f'{hurst_val:.2f}'}, +DI>{plus_val:.1f})")
    elif trending and (bear_aligned or minus_val > plus_val):
        regime      = "TRENDING_DOWN"
        hurst_boost = hurst_valid and hurst_val > 0.64 and adx_val < ADX_TREND_THRESHOLD
        if hurst_boost:
            base_conf = (hurst_val - 0.64) * 2
            adx_conf  = max(0, (adx_val - ADX_HURST_BOOST_THRESHOLD) / (ADX_TREND_THRESHOLD - ADX_HURST_BOOST_THRESHOLD))
            confidence = min(0.65, base_conf + adx_conf * 0.3)
            desc = (f"Trending down via Hurst persistence (ADX={adx_val:.1f} weak, "
                    f"Hurst={hurst_val:.2f} strong, -DI>{minus_val:.1f}) [Hurst-led]")
        else:
            base_conf  = (adx_val - ADX_TREND_THRESHOLD) / 25
            hurst_conf = (hurst_val - 0.5) * 2 if hurst_valid else 0
            confidence = min(1.0, base_conf + hurst_conf)
            desc       = (f"Trending down (ADX={adx_val:.1f}, "
                          f"Hurst={'N/A' if not hurst_valid else f'{hurst_val:.2f}'}, -DI>{minus_val:.1f})")
    else:
        regime     = "RANGING"
        confidence = min(1.0, max(0, ADX_TREND_THRESHOLD - adx_val) / ADX_TREND_THRESHOLD
                         + (max(0, 0.5 - hurst_val) * 4 if hurst_valid else 0))
        desc       = (f"Ranging (ADX={adx_val:.1f}, "
                      f"Hurst={'N/A' if not hurst_valid else f'{hurst_val:.2f}'})")

    # Penalise confidence when ADX is in the "weak" 20-25 band
    if ADX_WEAK_THRESHOLD < adx_val < ADX_TREND_THRESHOLD:
        confidence *= 0.6
        desc       += " [weak ADX — low confidence]"

    if not hurst_valid:
        desc += " [Hurst N/A — ADX-only]"

    confidence = round(max(0.0, min(1.0, confidence)), 2)

    log.debug(f"Regime: {regime} | ADX={adx_val:.1f} | Hurst={hurst_val:.2f}(valid={hurst_valid}) | "
              f"ER={er_val:.2f} | ATR%={atr_percentile:.0f} | conf={confidence:.2f}")

    return {
        "regime":           regime,
        "adx":              round(adx_val, 2),
        "plus_di":          round(plus_val, 2),
        "minus_di":         round(minus_val, 2),
        "hurst":            round(hurst_val, 3),
        "hurst_valid":      hurst_valid,
        "efficiency_ratio": round(er_val, 3),
        "atr_percentile":   atr_percentile,
        "confidence":       confidence,
        "description":      desc,
    }
