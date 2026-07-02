"""
Fractional Differentiation (FFD) — López de Prado, AFML Chapter 5.

Problem: Raw price series are non-stationary (unit root) → ML models learn
spurious patterns. Standard first-differencing (log returns) removes the unit
root but also destroys ALL long-memory (past prices stop being predictive).

Solution: Fractionally differentiate with d ∈ (0, 1). The FFD operator with
d* = min d such that ADF test rejects unit root preserves maximum memory while
achieving stationarity. Empirically d* ≈ 0.3–0.5 for most forex pairs.

Fixed-Width Window (FFW) implementation: uses only the last w coefficients to
avoid look-ahead bias. w=10–20 is stable for H1 bars; w→∞ gives the expanding
window version.

Typical usage:
    fracdiff_series = ffd_transform(df["close"], d=0.4, window=10)
    # Use fracdiff_series[-1] as a feature in the ML model
    # or find optimal d with find_min_d() on historical data

Integration in bot:
    - Compute FFD of close, RSI, and MACD series as extra ML features
    - Append to the 24-feature vector → richer feature set with memory preserved
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("FracDiff")


def _get_weights(d: float, size: int) -> np.ndarray:
    """
    Compute the FFD weight vector w for differentiation order d.
    w[0] = 1, w[k] = w[k-1] * (d - k + 1) / k
    Weights decay geometrically → finite window approximation is stable.
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w[::-1])   # reverse so w[-1] applies to most-recent obs


def ffd_transform(series: pd.Series,
                  d: float      = 0.4,
                  window: int   = 10,
                  threshold: float = 1e-5) -> pd.Series:
    """
    Fixed-Width Window Fractional Differentiation (FFD).

    Args:
        series:    Pandas Series of prices or indicator values.
        d:         Differentiation order ∈ (0, 1). d=0 = original, d=1 = log returns.
                   Use find_min_d() to find the minimum d that achieves stationarity.
        window:    Width of the weight window (truncation point). 10–20 for H1.
        threshold: Drop weights below this value to reduce noise (not used in FFD strict).

    Returns:
        Fractionally differentiated series, same index as input.
        First (window-1) values are NaN (warm-up period).
    """
    if d <= 0.0:
        return series.copy()
    if d >= 1.0:
        return np.log(series).diff()

    w    = _get_weights(d, window)
    vals = series.values.astype(float)
    n    = len(vals)
    result = np.full(n, np.nan)

    for t in range(window - 1, n):
        segment  = vals[t - window + 1: t + 1]
        # Apply log-transform before differencing (work in log-price space)
        log_seg  = np.log(segment)
        result[t] = float(np.dot(w, log_seg))

    return pd.Series(result, index=series.index)


def find_min_d(series: pd.Series,
               window: int = 10,
               d_range: tuple = (0.0, 1.0),
               steps: int    = 20,
               p_value: float = 0.05) -> float:
    """
    Find minimum d that makes the series stationary (ADF p-value < threshold).

    Implements the López de Prado approach:
      - Grid-search d in [d_min, d_max] with fine steps
      - For each d, run FFD and ADF test on the result
      - Return the smallest d where ADF p < p_value

    Args:
        series:   Price series (at least 200 bars for reliable ADF).
        window:   FFD window size.
        d_range:  (min_d, max_d) to search.
        steps:    Number of grid points.
        p_value:  ADF significance threshold.

    Returns:
        Optimal d (float). Returns 1.0 if no d < 1.0 achieves stationarity.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        log.debug("find_min_d: statsmodels not installed — using d=0.4 default")
        return 0.4

    if len(series) < 100:
        log.debug(f"find_min_d: series too short ({len(series)} bars) — using d=0.4 default")
        return 0.4

    d_grid = np.linspace(d_range[0], d_range[1], steps + 1)[1:]  # skip d=0 (non-stationary)

    for d in d_grid:
        fd = ffd_transform(series, d=float(d), window=window).dropna()
        if len(fd) < 30:
            continue
        try:
            adf_result = adfuller(fd.values, maxlag=1, regression="c", autolag=None)
            adf_pval   = adf_result[1]
            if adf_pval < p_value:
                log.info(f"FracDiff: min d={d:.2f} achieves stationarity (ADF p={adf_pval:.4f})")
                return float(d)
        except Exception:
            continue

    log.warning("FracDiff: no d < 1.0 achieves stationarity — check data quality")
    return 1.0


def fracdiff_features(df: pd.DataFrame,
                      d: float     = 0.4,
                      window: int  = 10) -> dict[str, float]:
    """
    Compute FFD-transformed features for the current bar.
    Returns a dict of scalar values suitable for appending to the ML feature vector.

    Features computed:
      fd_close:  FFD of close price (preserves long-memory in price)
      fd_range:  FFD of high-low range (volatility memory)

    Returns dict with 'fd_close', 'fd_range'. Values are 0.0 on insufficient data.
    """
    result = {"fd_close": 0.0, "fd_range": 0.0}
    try:
        if len(df) < window + 5:
            return result

        fd_close = ffd_transform(df["close"], d=d, window=window)
        fd_range = ffd_transform(df["high"] - df["low"], d=d, window=window)

        v_close = fd_close.iloc[-1]
        v_range = fd_range.iloc[-1]

        result["fd_close"] = float(v_close) if not np.isnan(v_close) else 0.0
        result["fd_range"] = float(v_range) if not np.isnan(v_range) else 0.0

    except Exception as e:
        log.debug(f"fracdiff_features error: {e}")

    return result
