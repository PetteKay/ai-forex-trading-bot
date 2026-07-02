"""
OHLC Volatility Estimators — efficient realized vol for H1 forex bars.

Yang-Zhang (2000) is the gold standard for OHLC data:
  - Handles overnight jumps (unlike Garman-Klass)
  - ~7x more efficient than close-to-close
  - Unbiased for drift
  - Best choice for H1 forex with 24h trading

Garman-Klass (1980): no overnight jump correction — better for continuous assets.
Parkinson (1980): high-low range only; simplest, good fallback.

Rolling percentile rank of vol → regime bucketing that adapts to pair-specific scale.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("VolEstimators")

_BARS_PER_YEAR_H1 = 24 * 252   # 24h × 252 trading days


def yang_zhang(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Yang-Zhang (2000) volatility — most efficient unbiased OHLC estimator.
    Handles opening jumps that Garman-Klass misses.

    Returns annualized vol series.
    """
    lo, hi, op, cl = df["low"], df["high"], df["open"], df["close"]
    cl_prev = cl.shift(1)

    log_oc  = np.log(op / cl_prev)      # overnight (open-vs-prev-close)
    log_co  = np.log(cl / op)           # close vs open of same bar
    log_ho  = np.log(hi / op)
    log_lo  = np.log(lo / op)

    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))

    sigma_o2  = log_oc.rolling(window).var(ddof=0)
    sigma_rs2 = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()
    sigma_c2  = log_co.rolling(window).var(ddof=0)

    sigma2 = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
    return np.sqrt(np.maximum(sigma2, 0.0) * _BARS_PER_YEAR_H1)


def garman_klass(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Garman-Klass (1980) — efficient for continuous 24/7 assets (XAUUSD, BTCUSD).
    Assumes no overnight gaps; slightly biased when gaps exist.
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    gk = (0.5 * log_hl ** 2 - (2.0 * np.log(2) - 1.0) * log_co ** 2).rolling(window).mean()
    return np.sqrt(np.maximum(gk, 0.0) * _BARS_PER_YEAR_H1)


def parkinson(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Parkinson (1980) — uses only high-low range.
    ~5x more efficient than close-to-close. Useful when open is unreliable.
    """
    log_hl = np.log(df["high"] / df["low"])
    pk = (log_hl ** 2 / (4.0 * np.log(2.0))).rolling(window).mean()
    return np.sqrt(np.maximum(pk, 0.0) * _BARS_PER_YEAR_H1)


def vol_percentile_rank(vol_series: pd.Series, window: int = 500) -> pd.Series:
    """
    Rolling percentile rank of realized vol (0–100).
    Adapts to pair-specific vol scale — no absolute threshold needed.
    Use this, not absolute vol, for regime bucketing.
    """
    min_obs = max(50, window // 5)
    return vol_series.rolling(window, min_periods=min_obs).rank(pct=True) * 100.0


def vol_of_vol(vol_series: pd.Series, window: int = 20) -> pd.Series:
    """
    Volatility-of-volatility (VoV) — measures regime instability.
    High VoV = transitioning regime → tighten position sizes, widen SL.
    Low VoV  = stable regime → normal sizing.
    """
    return vol_series.rolling(window).std()


def regime_from_vol(df: pd.DataFrame,
                    window_est: int  = 20,
                    window_rank: int = 500) -> dict:
    """
    Single-call interface used in bot.py.
    Returns current bar's vol regime as a dict.

    Regime labels:
      QUIET    — < 25th pct   (low vol, often consolidation)
      NORMAL   — 25–75th pct  (standard conditions)
      ELEVATED — 75–90th pct  (elevated risk, widen SL)
      EXTREME  — > 90th pct   (event risk, reduce size sharply)
    """
    if len(df) < window_est + 5:
        return {
            "yz_vol": 0.0, "vol_pct_rank": 50.0,
            "vol_regime_label": "NORMAL", "vol_of_vol": 0.0,
        }

    try:
        yz      = yang_zhang(df, window=window_est)
        rank    = vol_percentile_rank(yz, window=window_rank)
        vov     = vol_of_vol(yz, window=window_est)

        cur_vol  = float(yz.iloc[-1])  if not np.isnan(yz.iloc[-1])   else 0.0
        cur_rank = float(rank.iloc[-1]) if not np.isnan(rank.iloc[-1]) else 50.0
        cur_vov  = float(vov.iloc[-1])  if not np.isnan(vov.iloc[-1])  else 0.0

        if cur_rank < 25:
            label = "QUIET"
        elif cur_rank < 75:
            label = "NORMAL"
        elif cur_rank < 90:
            label = "ELEVATED"
        else:
            label = "EXTREME"

        return {
            "yz_vol":          round(cur_vol, 6),
            "vol_pct_rank":    round(cur_rank, 1),
            "vol_regime_label": label,
            "vol_of_vol":      round(cur_vov, 6),
        }

    except Exception as e:
        log.debug(f"Vol regime error: {e}")
        return {
            "yz_vol": 0.0, "vol_pct_rank": 50.0,
            "vol_regime_label": "NORMAL", "vol_of_vol": 0.0,
        }
