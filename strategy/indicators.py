"""
Full indicator suite: EMA, RSI, ATR, MACD, Bollinger Bands, market structure,
candlestick patterns, volume analysis, and multi-timeframe confluence.
"""

import pandas as pd
import numpy as np
from datetime import datetime, UTC


# ── Core indicator helpers ───────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _macd(series: pd.Series, fast: int, slow: int, signal: int):
    macd_line   = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series, period: int, std: float):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std * sigma, mid, mid - std * sigma


def _market_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    recent = df.iloc[-lookback:]
    if (recent["high"].iloc[-1] > recent["high"].iloc[0] and
            recent["low"].iloc[-1] > recent["low"].iloc[0]):
        return "UPTREND"
    if (recent["high"].iloc[-1] < recent["high"].iloc[0] and
            recent["low"].iloc[-1] < recent["low"].iloc[0]):
        return "DOWNTREND"
    return "SIDEWAYS"


def _trading_session() -> list[str]:
    hour = datetime.now(UTC).hour
    sessions = []
    if 22 <= hour or hour < 8:
        sessions.append("ASIAN")
    if 8 <= hour < 16:
        sessions.append("LONDON")
    if 13 <= hour < 22:
        sessions.append("NEW_YORK")
    return sessions or ["OFF_HOURS"]


# ── Candlestick pattern detection ────────────────────────────────────────────

def _candlestick_patterns(df: pd.DataFrame) -> list[str]:
    """
    Detect ~65 candlestick patterns (Steve Nison complete set) from the last 5 bars.
    Covers single-candle, two-candle, three-candle, and five-candle formations.
    """
    if len(df) < 3:
        return []

    patterns = []

    # ── Bar references ────────────────────────────────────────────────────────
    c  = df.iloc[-1]
    p  = df.iloc[-2]
    pp = df.iloc[-3]

    o,   h,   l,   cl   = float(c["open"]),  float(c["high"]),  float(c["low"]),  float(c["close"])
    po,  ph,  pl,  pcl  = float(p["open"]),  float(p["high"]),  float(p["low"]),  float(p["close"])
    ppo, pph, ppl, ppcl = float(pp["open"]), float(pp["high"]), float(pp["low"]), float(pp["close"])

    body      = abs(cl  - o)
    body_prev = abs(pcl - po)
    body_pp   = abs(ppcl - ppo)
    range_c   = h  - l
    range_p   = ph - pl
    range_pp  = pph - ppl

    bull_c  = cl > o;    bear_c  = cl < o
    bull_p  = pcl > po;  bear_p  = pcl < po
    bull_pp = ppcl > ppo; bear_pp = ppcl < ppo

    upper_wick   = h  - max(o,  cl)
    lower_wick   = min(o,  cl)  - l
    upper_wick_p = ph - max(po, pcl)
    lower_wick_p = min(po, pcl) - pl

    midpoint_p = (po + pcl) / 2
    tol        = max(range_c, range_p) * 0.05
    tol_open   = max(range_c, range_p) * 0.02

    is_doji_c  = range_c  > 0 and body      / range_c  < 0.1
    is_doji_p  = range_p  > 0 and body_prev / range_p  < 0.1
    is_doji_pp = range_pp > 0 and body_pp   / range_pp < 0.1

    prev_bear_maru = range_p > 0 and body_prev / range_p > 0.85 and bear_p
    prev_bull_maru = range_p > 0 and body_prev / range_p > 0.85 and bull_p
    curr_bear_maru = range_c > 0 and body      / range_c > 0.85 and bear_c
    curr_bull_maru = range_c > 0 and body      / range_c > 0.85 and bull_c

    # Average body over last 10 bars for Long Day detection
    recent_bodies = [abs(float(df.iloc[i]["close"]) - float(df.iloc[i]["open"]))
                     for i in range(max(-len(df), -11), -1)]
    avg_body = sum(recent_bodies) / len(recent_bodies) if recent_bodies else body

    # ═══════════════════════════════════════════════════════════════════
    # SINGLE-CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════════════

    # Four-Price Doji (open = high = low = close — flat bar)
    if range_c < max(abs(o), 1e-10) * 0.0001:
        patterns.append("FOUR_PRICE_DOJI")

    if range_c > 0:
        body_r = body / range_c

        # ── Doji family ──────────────────────────────────────────────────────
        if body_r < 0.1:
            if upper_wick > range_c * 0.35 and lower_wick > range_c * 0.35:
                patterns.append("LONG_LEGGED_DOJI")
            elif lower_wick < range_c * 0.05 and upper_wick > range_c * 0.6:
                patterns.append("GRAVESTONE_DOJI")
            elif upper_wick < range_c * 0.05 and lower_wick > range_c * 0.6:
                patterns.append("DRAGONFLY_DOJI")
            elif abs(upper_wick - lower_wick) < range_c * 0.1 and upper_wick > range_c * 0.3:
                patterns.append("RICKSHAW_MAN")
            else:
                patterns.append("DOJI")

        # ── Spinning Top / High Wave ─────────────────────────────────────────
        if (0.1 <= body_r <= 0.35
                and upper_wick > range_c * 0.15
                and lower_wick > range_c * 0.15):
            if upper_wick > range_c * 0.35 and lower_wick > range_c * 0.35:
                patterns.append("HIGH_WAVE")
            else:
                patterns.append("SPINNING_TOP")

        # ── Long Day ─────────────────────────────────────────────────────────
        if avg_body > 0 and body > avg_body * 1.5 and body_r > 0.55:
            patterns.append("LONG_WHITE_DAY" if bull_c else "LONG_BLACK_DAY")

        if body > 0:
            # ── Hammer / Hanging Man ──────────────────────────────────────────
            if lower_wick >= 2 * body and upper_wick < body * 0.5:
                patterns.append("HAMMER" if bear_p else "HANGING_MAN")
            # ── Shooting Star / Inverted Hammer ───────────────────────────────
            if upper_wick >= 2 * body and lower_wick < body * 0.5:
                patterns.append("SHOOTING_STAR" if bull_p else "INVERTED_HAMMER")

        # ── Marubozu ─────────────────────────────────────────────────────────
        if body_r > 0.85:
            patterns.append("BULL_MARUBOZU" if bull_c else "BEAR_MARUBOZU")

        # ── Belt Hold ────────────────────────────────────────────────────────
        if body > 0 and body_r > 0.6:
            if bull_c and (o - l) < range_c * 0.05:
                patterns.append("BULLISH_BELT_HOLD")
            if bear_c and (h - o) < range_c * 0.05:
                patterns.append("BEARISH_BELT_HOLD")

        # ── Pin Bar ──────────────────────────────────────────────────────────
        if body_r < 0.3:
            if lower_wick > upper_wick * 2:
                patterns.append("BULLISH_PIN_BAR")
            elif upper_wick > lower_wick * 2:
                patterns.append("BEARISH_PIN_BAR")

    # ═══════════════════════════════════════════════════════════════════
    # TWO-CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════════════

    # ── Engulfing ────────────────────────────────────────────────────────────
    if bear_c and bull_p and o >= pcl and cl <= po and body >= body_prev:
        patterns.append("BEARISH_ENGULFING")
    if bull_c and bear_p and o <= pcl and cl >= po and body >= body_prev:
        patterns.append("BULLISH_ENGULFING")

    # ── Harami ───────────────────────────────────────────────────────────────
    if bear_p and bull_c and o > pcl and cl < po and body < body_prev * 0.5:
        patterns.append("BULLISH_HARAMI")
    if bull_p and bear_c and o < pcl and cl > po and body < body_prev * 0.5:
        patterns.append("BEARISH_HARAMI")

    # Harami Cross (middle is a doji)
    if bear_p and is_doji_c and o > pcl and o < po:
        patterns.append("BULLISH_HARAMI_CROSS")
    if bull_p and is_doji_c and o > po and o < pcl:
        patterns.append("BEARISH_HARAMI_CROSS")

    # ── Piercing / Dark Cloud ────────────────────────────────────────────────
    if bear_p and bull_c and o < pcl and cl > midpoint_p and cl < po:
        patterns.append("PIERCING_LINE")
    if bull_p and bear_c and o > pcl and cl < midpoint_p and cl > po:
        patterns.append("DARK_CLOUD_COVER")

    # ── On Neck / In Neck / Thrusting (bearish continuation below midpoint) ──
    if bear_p and bull_c and o < pl:
        if abs(cl - pcl) < tol:
            patterns.append("ON_NECK_LINE")
        elif pcl < cl < pcl + range_p * 0.15:
            patterns.append("IN_NECK_LINE")
        elif o < pcl < cl < midpoint_p:
            patterns.append("THRUSTING_PATTERN")

    # ── Meeting Lines / Counterattack ────────────────────────────────────────
    meet_tol = max(range_c, range_p) * 0.03
    if (bear_p and bull_c and abs(cl - pcl) < meet_tol
            and body_prev > range_p * 0.5 and body > range_c * 0.5):
        patterns.append("BULLISH_COUNTERATTACK")
    if (bull_p and bear_c and abs(cl - pcl) < meet_tol
            and body_prev > range_p * 0.5 and body > range_c * 0.5):
        patterns.append("BEARISH_COUNTERATTACK")

    # ── Kicking (opposite marubozus with gap) ────────────────────────────────
    if prev_bear_maru and curr_bull_maru and o > ph:
        patterns.append("BULLISH_KICKING")
    if prev_bull_maru and curr_bear_maru and o < pl:
        patterns.append("BEARISH_KICKING")

    # ── Separating Lines (same open, opposite direction) ─────────────────────
    if bear_p and bull_c and abs(o - po) < tol_open:
        patterns.append("BULLISH_SEPARATING_LINES")
    if bull_p and bear_c and abs(o - po) < tol_open:
        patterns.append("BEARISH_SEPARATING_LINES")

    # ── Tweezer ──────────────────────────────────────────────────────────────
    if bull_p and bear_c and abs(h - ph) < tol and upper_wick > 0 and upper_wick_p > 0:
        patterns.append("TWEEZER_TOP")
    if bear_p and bull_c and abs(l - pl) < tol and lower_wick > 0 and lower_wick_p > 0:
        patterns.append("TWEEZER_BOTTOM")

    # ── Windows / Gaps ───────────────────────────────────────────────────────
    if l > ph:
        patterns.append("RISING_WINDOW")
    if h < pl:
        patterns.append("FALLING_WINDOW")

    # ═══════════════════════════════════════════════════════════════════
    # THREE-CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════════════

    # ── Stars (Morning / Evening — doji and non-doji middle) ─────────────────
    if bear_pp and body_prev < body_pp * 0.5 and bull_c and cl > (ppo + ppcl) / 2:
        patterns.append("MORNING_DOJI_STAR" if is_doji_p else "MORNING_STAR")
    if bull_pp and body_prev < body_pp * 0.5 and bear_c and cl < (ppo + ppcl) / 2:
        patterns.append("EVENING_DOJI_STAR" if is_doji_p else "EVENING_STAR")

    # ── Tri-Star (3 consecutive dojis — supreme indecision) ──────────────────
    if is_doji_c and is_doji_p and is_doji_pp:
        patterns.append("BULLISH_TRI_STAR" if bear_pp else "BEARISH_TRI_STAR")

    # ── Three White Soldiers ─────────────────────────────────────────────────
    if (bull_c and bull_p and bull_pp
            and cl > pcl > ppcl
            and o > po and o < pcl
            and po > ppo and po < ppcl
            and upper_wick < body * 0.3
            and upper_wick_p < body_prev * 0.3):
        patterns.append("THREE_WHITE_SOLDIERS")

    # ── Three Black Crows ────────────────────────────────────────────────────
    if (bear_c and bear_p and bear_pp
            and cl < pcl < ppcl
            and o < po and o > pcl
            and po < ppo and po > ppcl
            and lower_wick < body * 0.3
            and lower_wick_p < body_prev * 0.3):
        patterns.append("THREE_BLACK_CROWS")

    # ── Identical Three Crows (each opens exactly at prior close) ────────────
    if (bear_c and bear_p and bear_pp
            and cl < pcl < ppcl
            and abs(o - pcl) < tol
            and abs(po - ppcl) < tol):
        patterns.append("IDENTICAL_THREE_CROWS")

    # ── Three Inside Up / Down ───────────────────────────────────────────────
    if (bear_pp and bull_p and bull_c
            and po > ppcl and pcl < ppo
            and body_prev < body_pp * 0.5
            and cl > ppo):
        patterns.append("THREE_INSIDE_UP")
    if (bull_pp and bear_p and bear_c
            and po < ppcl and pcl > ppo
            and body_prev < body_pp * 0.5
            and cl < ppo):
        patterns.append("THREE_INSIDE_DOWN")

    # ── Three Outside Up / Down ──────────────────────────────────────────────
    if (bear_pp
            and bull_p and po <= ppcl and pcl >= ppo
            and bull_c and cl > pcl):
        patterns.append("THREE_OUTSIDE_UP")
    if (bull_pp
            and bear_p and po >= ppcl and pcl <= ppo
            and bear_c and cl < pcl):
        patterns.append("THREE_OUTSIDE_DOWN")

    # ── Abandoned Baby (gaps required — rare in forex) ───────────────────────
    if is_doji_p:
        if bear_pp and bull_c and ph < ppcl and l > ph:
            patterns.append("BULLISH_ABANDONED_BABY")
        if bull_pp and bear_c and pl > ppcl and h < pl:
            patterns.append("BEARISH_ABANDONED_BABY")

    # ── Tasuki Gaps (continuation through a gap) ─────────────────────────────
    if (bull_pp and bull_p and pl > pph          # gap up between pp and p
            and bear_c
            and o >= pl and o <= ph
            and cl > pph and cl < pl):           # closes in gap but can't fill it
        patterns.append("UPSIDE_TASUKI_GAP")
    if (bear_pp and bear_p and ph < ppl          # gap down between pp and p
            and bull_c
            and o >= pl and o <= ph
            and cl < ppl and cl > ph):           # closes in gap but can't fill it
        patterns.append("DOWNSIDE_TASUKI_GAP")

    # ── Advance Block (3 bulls weakening — shrinking bodies, growing upper wicks)
    if (bull_c and bull_p and bull_pp
            and cl > pcl > ppcl
            and body < body_prev
            and upper_wick > body * 0.3
            and upper_wick_p > body_prev * 0.2):
        patterns.append("ADVANCE_BLOCK")

    # ── Deliberation (3 bulls, 3rd tiny — momentum exhaustion) ───────────────
    if (bull_c and bull_p and bull_pp
            and cl > pcl > ppcl
            and body < body_prev * 0.3):
        patterns.append("DELIBERATION")

    # ── Stick Sandwich (bear–bull–bear, outer closes match — bullish reversal)
    if bear_pp and bull_p and bear_c and abs(cl - ppcl) < tol:
        patterns.append("STICK_SANDWICH")

    # ── Upside Gap Two Crows (bearish) ───────────────────────────────────────
    if (bull_pp
            and bear_p and po > ppcl            # day2 gaps up above day1 close
            and bear_c and o > pcl and o < po   # day3 opens inside day2 body
            and cl <= ppcl):                    # day3 closes into day1 body
        patterns.append("UPSIDE_GAP_TWO_CROWS")

    # ── Two Crows (bearish) ───────────────────────────────────────────────────
    if (bull_pp
            and bear_p and po > ppcl            # day2 gaps up, bearish
            and bear_c and o > po               # day3 opens above day2 open
            and cl >= ppo and cl <= ppcl):      # day3 closes within day1 body
        patterns.append("TWO_CROWS")

    # ═══════════════════════════════════════════════════════════════════
    # FIVE-CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════════════

    if len(df) >= 5:
        b4 = df.iloc[-5]
        b3 = df.iloc[-4]

        b4o, b4c = float(b4["open"]), float(b4["close"])
        b4h, b4l = float(b4["high"]), float(b4["low"])
        b3o, b3c = float(b3["open"]), float(b3["close"])
        b3h, b3l = float(b3["high"]), float(b3["low"])
        b4_body  = abs(b4c - b4o)
        b3_body  = abs(b3c - b3o)

        # Rising Three Methods: big bull → 3 small bears within body → big bull
        if (b4c > b4o
                and b3c < b3o and ppcl < ppo and pcl < po
                and bull_c and cl > b4c
                and b3l > b4o and b3h < b4c
                and ppl > b4o and pph < b4c
                and pl  > b4o and ph  < b4c
                and b3_body < b4_body * 0.5):
            patterns.append("RISING_THREE_METHODS")

        # Falling Three Methods: big bear → 3 small bulls within body → big bear
        if (b4c < b4o
                and b3c > b3o and ppcl > ppo and pcl > po
                and bear_c and cl < b4c
                and b3h < b4o and b3l > b4c
                and pph < b4o and ppl > b4c
                and ph  < b4o and pl  > b4c
                and b3_body < b4_body * 0.5):
            patterns.append("FALLING_THREE_METHODS")

    return patterns


# ── Volume analysis ──────────────────────────────────────────────────────────

def _volume_analysis(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Analyse volume relative to recent average, including volume-price divergence."""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return {"rel_volume": 1.0, "signal": "NO_DATA", "climax": False,
                "vol_price_div": "NONE"}

    recent_vol = df["volume"].iloc[-lookback:]
    avg_vol    = float(recent_vol.mean())
    curr_vol   = float(df["volume"].iloc[-1])
    rel_vol    = curr_vol / avg_vol if avg_vol > 0 else 1.0

    climax = rel_vol > 3.0

    if rel_vol > 2.0:
        signal = "HIGH_VOLUME"
    elif rel_vol > 1.3:
        signal = "ABOVE_AVERAGE"
    elif rel_vol < 0.7:
        signal = "LOW_VOLUME"
    else:
        signal = "AVERAGE"

    # Volume-price divergence
    curr_bull = float(df["close"].iloc[-1]) > float(df["close"].iloc[-2])
    curr_bear = float(df["close"].iloc[-1]) < float(df["close"].iloc[-2])
    if curr_bull and signal == "HIGH_VOLUME":
        vol_price_div = "STRONG_BULL"   # price up + high volume = confirmed
    elif curr_bear and signal == "HIGH_VOLUME":
        vol_price_div = "STRONG_BEAR"   # price down + high volume = confirmed
    elif curr_bull and signal == "LOW_VOLUME":
        vol_price_div = "WEAK_BULL"     # price up + low volume = unconvincing
    elif curr_bear and signal == "LOW_VOLUME":
        vol_price_div = "WEAK_BEAR"     # price down + low volume = unconvincing
    else:
        vol_price_div = "NONE"

    return {
        "rel_volume":   round(rel_vol, 2),
        "signal":       signal,
        "climax":       climax,
        "vol_price_div": vol_price_div,
    }


# ── Stochastic oscillator ────────────────────────────────────────────────────

def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
    """Stochastic %K and %D. Returns (k, d) — both in 0-100 range."""
    if len(df) < k_period:
        return 50.0, 50.0
    low_min  = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k_series = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    d_series = k_series.rolling(d_period).mean()
    k = float(k_series.iloc[-1]) if not pd.isna(k_series.iloc[-1]) else 50.0
    d = float(d_series.iloc[-1]) if not pd.isna(d_series.iloc[-1]) else 50.0
    return round(k, 1), round(d, 1)


# ── Higher timeframe bias ────────────────────────────────────────────────────

def _htf_bias(htf_df: pd.DataFrame) -> str:
    """
    Compute a simple directional bias from a higher timeframe DataFrame.
    Returns: BULLISH | BEARISH | NEUTRAL
    """
    if htf_df is None or len(htf_df) < 5:
        return "NEUTRAL"

    close  = htf_df["close"]
    ema21  = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    price  = float(close.iloc[-1])

    if price > ema21 > ema200:
        return "BULLISH"
    if price < ema21 < ema200:
        return "BEARISH"
    return "NEUTRAL"


# ── Main compute function ────────────────────────────────────────────────────

def compute(df: pd.DataFrame, cfg, htf_df: pd.DataFrame = None, symbol: str = "") -> dict:
    """
    Compute all indicators and return a structured context dict.
    Optionally accepts a higher-timeframe DataFrame for HTF bias.
    Pass symbol explicitly when scanning multiple instruments so the returned
    context contains the correct symbol name rather than cfg.SYMBOL default.
    """
    df = df.copy()

    df["ema_fast"]  = _ema(df["close"], cfg.EMA_FAST)
    df["ema_slow"]  = _ema(df["close"], cfg.EMA_SLOW)
    df["ema_trend"] = _ema(df["close"], cfg.EMA_TREND)
    df["rsi"]       = _rsi(df["close"], cfg.RSI_PERIOD)
    df["atr"]       = _atr(df, cfg.ATR_PERIOD)
    df["macd"], df["macd_sig"], df["macd_hist"] = _macd(
        df["close"], cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL
    )
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bollinger(
        df["close"], cfg.BB_PERIOD, cfg.BB_STD
    )

    df.dropna(inplace=True)
    if len(df) < 5:
        return {}

    curr  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = float(curr["close"])

    ema_cross_up    = (float(prev["ema_fast"]) <= float(prev["ema_slow"]) and
                       float(curr["ema_fast"]) > float(curr["ema_slow"]))
    ema_cross_down  = (float(prev["ema_fast"]) >= float(prev["ema_slow"]) and
                       float(curr["ema_fast"]) < float(curr["ema_slow"]))
    macd_cross_up   = float(prev["macd_hist"]) <= 0 and float(curr["macd_hist"]) > 0
    macd_cross_down = float(prev["macd_hist"]) >= 0 and float(curr["macd_hist"]) < 0
    bb_pct          = (price - float(curr["bb_lower"])) / (
        float(curr["bb_upper"]) - float(curr["bb_lower"]) + 1e-10
    )

    atr_series  = df["atr"].iloc[-50:]
    atr_pct     = float((atr_series <= float(curr["atr"])).mean() * 100)
    structure   = _market_structure(df)
    sessions    = _trading_session()
    patterns    = _candlestick_patterns(df)
    vol_data    = _volume_analysis(df)
    htf_b       = _htf_bias(htf_df)
    stoch_k, stoch_d = _stochastic(df)

    recent_candles = [
        {
            "time":      str(df.index[i]),
            "open":      round(float(df.iloc[i]["open"]), 5),
            "high":      round(float(df.iloc[i]["high"]), 5),
            "low":       round(float(df.iloc[i]["low"]), 5),
            "close":     round(float(df.iloc[i]["close"]), 5),
            "direction": "UP" if float(df.iloc[i]["close"]) > float(df.iloc[i]["open"]) else "DOWN",
        }
        for i in range(-4, 0)
    ]

    # RSI + MACD divergence — swing-based detection over last 20 bars
    _rsi_s   = df["rsi"].iloc[-20:].values
    _macd_s  = df["macd_hist"].iloc[-20:].values
    _price_s = df["close"].iloc[-20:].values
    n20      = len(_price_s)

    def _swing_highs(arr, n):
        return [i for i in range(2, n - 2)
                if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > arr[i-2]]

    def _swing_lows(arr, n):
        return [i for i in range(2, n - 2)
                if arr[i] < arr[i-1] and arr[i] < arr[i+1] and arr[i] < arr[i-2]]

    _ph = _swing_highs(_price_s, n20)
    _pl = _swing_lows(_price_s, n20)

    rsi_bearish_div  = False
    rsi_bullish_div  = False
    macd_bearish_div = False
    macd_bullish_div = False

    if len(_ph) >= 2:
        h1, h2 = _ph[-2], _ph[-1]
        if _price_s[h2] > _price_s[h1]:
            if _rsi_s[h2]  < _rsi_s[h1]:  rsi_bearish_div  = True
            if _macd_s[h2] < _macd_s[h1]: macd_bearish_div = True
    if len(_pl) >= 2:
        l1, l2 = _pl[-2], _pl[-1]
        if _price_s[l2] < _price_s[l1]:
            if _rsi_s[l2]  > _rsi_s[l1]:  rsi_bullish_div  = True
            if _macd_s[l2] > _macd_s[l1]: macd_bullish_div = True

    # Fallback to simple 5-bar comparison when no swings detected
    if not _ph and not _pl and n20 >= 6:
        rsi_bearish_div = _price_s[-1] > _price_s[-6] and _rsi_s[-1] < _rsi_s[-6]
        rsi_bullish_div = _price_s[-1] < _price_s[-6] and _rsi_s[-1] > _rsi_s[-6]

    bearish_div = rsi_bearish_div
    bullish_div = rsi_bullish_div

    return {
        "price":             round(price, 5),
        "symbol":            symbol or cfg.SYMBOL,
        "timeframe":         cfg.TIMEFRAME,

        "ema_fast":          round(float(curr["ema_fast"]), 5),
        "ema_slow":          round(float(curr["ema_slow"]), 5),
        "ema_trend":         round(float(curr["ema_trend"]), 5),
        "price_above_fast":  price > float(curr["ema_fast"]),
        "price_above_slow":  price > float(curr["ema_slow"]),
        "price_above_trend": price > float(curr["ema_trend"]),
        "fast_above_slow":   float(curr["ema_fast"]) > float(curr["ema_slow"]),
        "ema_cross_up":      ema_cross_up,
        "ema_cross_down":    ema_cross_down,

        "rsi":               round(float(curr["rsi"]), 1),
        "rsi_state":         ("OVERBOUGHT" if float(curr["rsi"]) > 70
                              else "OVERSOLD" if float(curr["rsi"]) < 30 else "NEUTRAL"),
        "rsi_rising":        float(curr["rsi"]) > float(prev["rsi"]),
        "rsi_divergence":    ("BEARISH" if bearish_div else "BULLISH" if bullish_div else "NONE"),
        "macd_divergence":   ("BEARISH" if macd_bearish_div else "BULLISH" if macd_bullish_div else "NONE"),

        "stoch_k":           stoch_k,
        "stoch_d":           stoch_d,
        "stoch_state":       ("OVERBOUGHT" if stoch_k > 80 else "OVERSOLD" if stoch_k < 20 else "NEUTRAL"),
        "stoch_cross_up":    stoch_k > stoch_d and stoch_k < 30,   # bullish cross in oversold
        "stoch_cross_down":  stoch_k < stoch_d and stoch_k > 70,   # bearish cross in overbought

        "macd":              round(float(curr["macd"]), 6),
        "macd_signal":       round(float(curr["macd_sig"]), 6),
        "macd_histogram":    round(float(curr["macd_hist"]), 6),
        "macd_cross_up":     macd_cross_up,
        "macd_cross_down":   macd_cross_down,
        "macd_positive":     float(curr["macd_hist"]) > 0,

        "bb_upper":          round(float(curr["bb_upper"]), 5),
        "bb_middle":         round(float(curr["bb_mid"]), 5),
        "bb_lower":          round(float(curr["bb_lower"]), 5),
        "bb_pct":            round(bb_pct, 2),
        "bb_squeeze":        (float(curr["bb_upper"]) - float(curr["bb_lower"])) / price < 0.005,

        "atr":               round(float(curr["atr"]), 5),
        "atr_percentile":    round(atr_pct, 0),

        "market_structure":  structure,
        "sessions":          sessions,
        "recent_candles":    recent_candles,
        "candlestick_patterns": patterns,
        "volume_analysis":   vol_data,
        "htf_bias":          htf_b,

        "_df": df,
    }


# ── Market fingerprint ───────────────────────────────────────────────────────

def market_fingerprint(ctx: dict) -> str:
    """
    Encode the market state as a compact string for memory similarity lookup.
    10 characters: trend/ema/rsi/macd/structure/rsi-state/htf/volume/stoch/divergence.
    """
    rsi = ctx.get("rsi", 50)
    parts = [
        "T" if ctx.get("price_above_trend") else "t",
        "F" if ctx.get("fast_above_slow") else "f",
        "R" if ctx.get("rsi_rising") else "r",
        "M" if ctx.get("macd_positive") else "m",
        ctx.get("market_structure", "?")[0],
        ctx.get("rsi_state", "?")[0],
        "H" if ctx.get("htf_bias") == "BULLISH" else "h" if ctx.get("htf_bias") == "BEARISH" else "n",
        "V" if ctx.get("volume_analysis", {}).get("signal") == "HIGH_VOLUME" else "v",
        "S" if ctx.get("stoch_state") == "OVERSOLD" else "s" if ctx.get("stoch_state") == "OVERBOUGHT" else "n",
        "D" if ctx.get("rsi_divergence") != "NONE" else "d",
    ]
    return "".join(parts)


# ── Feature vector for embedding-based similarity ────────────────────────────

def feature_vector(ctx: dict) -> list[float]:
    """
    Encode market state as a normalized float vector for cosine similarity.
    Used by the embedding memory store for richer recall.
    """
    rsi = ctx.get("rsi", 50)
    return [
        1.0 if ctx.get("price_above_trend") else 0.0,
        1.0 if ctx.get("fast_above_slow") else 0.0,
        1.0 if ctx.get("rsi_rising") else 0.0,
        1.0 if ctx.get("macd_positive") else 0.0,
        rsi / 100.0,
        ctx.get("bb_pct", 0.5),
        ctx.get("atr_percentile", 50) / 100.0,
        1.0 if ctx.get("ema_cross_up") else 0.0,
        1.0 if ctx.get("ema_cross_down") else 0.0,
        1.0 if ctx.get("bb_squeeze") else 0.0,
        1.0 if ctx.get("market_structure") == "UPTREND" else 0.0,
        1.0 if ctx.get("market_structure") == "DOWNTREND" else 0.0,
        1.0 if ctx.get("htf_bias") == "BULLISH" else 0.0,
        1.0 if ctx.get("htf_bias") == "BEARISH" else 0.0,
        1.0 if ctx.get("rsi_divergence") == "BEARISH" else 0.0,
        1.0 if ctx.get("rsi_divergence") == "BULLISH" else 0.0,
        ctx.get("stoch_k", 50) / 100.0,
        1.0 if ctx.get("stoch_state") == "OVERSOLD"   else 0.0,
        1.0 if ctx.get("stoch_state") == "OVERBOUGHT" else 0.0,
        1.0 if ctx.get("macd_divergence") == "BEARISH" else 0.0,
        1.0 if ctx.get("macd_divergence") == "BULLISH" else 0.0,
    ]
