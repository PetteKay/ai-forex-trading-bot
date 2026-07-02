"""
Numba-JIT-accelerated Hurst exponent (R/S analysis).

Provides ~10-50x speedup over the pure-numpy loop.
Falls back transparently to numpy if numba is not installed.

Install numba for acceleration: pip install numba

This module exports a single function:
    compute_hurst_fast(series, min_lag, max_lag) -> float

The caller (strategy/regime.py) handles the minimum-bar guard and
returns the (value, reliable) tuple — this module only does the math.
"""
import numpy as np
import logging

log = logging.getLogger("FastHurst")

# ── Numpy reference implementation (always available) ─────────────────────────

def _hurst_numpy(series: np.ndarray, min_lag: int, max_lag: int) -> float:
    n         = len(series)
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
            s          = np.std(chunk, ddof=1)
            if s < 1e-10:
                continue
            rs_list.append(r / s)
        if rs_list:
            rs_values.append(float(np.mean(rs_list)))

    if len(rs_values) < 3:
        return 0.5

    log_lags = np.log(list(lags[: len(rs_values)]))
    log_rs   = np.log(rs_values)
    try:
        h, _ = np.polyfit(log_lags, log_rs, 1)
        return float(np.clip(h, 0.0, 1.0))
    except Exception:
        return 0.5


# ── Numba JIT implementation (optional) ───────────────────────────────────────

try:
    from numba import njit as _njit

    @_njit(cache=True, fastmath=True)
    def _rs_hurst_jit(series: np.ndarray, min_lag: int, max_lag: int) -> float:
        """
        Full R/S Hurst in a single JIT-compiled kernel.
        Called once with the whole series — no Python-level per-lag loop overhead.
        """
        n           = len(series)
        actual_max  = min(max_lag, n // 2)
        if actual_max <= min_lag:
            return 0.5

        n_lags  = actual_max - min_lag
        log_lags = np.empty(n_lags)
        log_rs   = np.empty(n_lags)
        valid    = 0

        for k in range(n_lags):
            lag     = min_lag + k
            n_chunks = (n - lag) // lag
            if n_chunks == 0:
                continue

            rs_sum   = 0.0
            rs_count = 0

            for ci in range(n_chunks):
                start = ci * lag
                # Mean
                mean = 0.0
                for j in range(lag):
                    mean += series[start + j]
                mean /= lag

                # Cumulative deviation — track range
                cum  = 0.0
                cmax = -1e18
                cmin =  1e18
                for j in range(lag):
                    cum += series[start + j] - mean
                    if cum > cmax:
                        cmax = cum
                    if cum < cmin:
                        cmin = cum

                r = cmax - cmin
                if r == 0.0:
                    continue

                # Standard deviation (population)
                var = 0.0
                for j in range(lag):
                    d   = series[start + j] - mean
                    var += d * d
                std = (var / lag) ** 0.5
                if std < 1e-10:
                    continue

                rs_sum   += r / std
                rs_count += 1

            if rs_count > 0:
                log_lags[valid] = np.log(float(lag))
                log_rs[valid]   = np.log(rs_sum / rs_count)
                valid          += 1

        if valid < 3:
            return 0.5

        # OLS slope of log(R/S) vs log(lag) → Hurst
        x = log_lags[:valid]
        y = log_rs[:valid]
        sx, sy, sxy, sx2 = 0.0, 0.0, 0.0, 0.0
        for i in range(valid):
            sx  += x[i]
            sy  += y[i]
            sxy += x[i] * y[i]
            sx2 += x[i] * x[i]

        denom = valid * sx2 - sx * sx
        if abs(denom) < 1e-10:
            return 0.5

        h = (valid * sxy - sx * sy) / denom
        if h < 0.0:
            return 0.0
        if h > 1.0:
            return 1.0
        return h

    def compute_hurst_fast(series: np.ndarray, min_lag: int = 2, max_lag: int = 50) -> float:
        """Numba-JIT Hurst — ~10-50x faster than the numpy loop for large series."""
        try:
            return float(_rs_hurst_jit(series, min_lag, max_lag))
        except Exception as e:
            log.debug(f"Numba Hurst error ({e}); using numpy fallback")
            return _hurst_numpy(series, min_lag, max_lag)

    log.debug("FastHurst: Numba JIT loaded — accelerated R/S Hurst active")

except ImportError:
    log.debug("FastHurst: numba not installed — using numpy (pip install numba for 10-50x speedup)")

    def compute_hurst_fast(series: np.ndarray, min_lag: int = 2, max_lag: int = 50) -> float:
        return _hurst_numpy(series, min_lag, max_lag)
