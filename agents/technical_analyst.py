"""
Technical Analyst Agent — builds the bull case from all indicator data.

Reads the full indicator context (primary + higher timeframe, candlestick
patterns, volume, divergences, regime) and produces a structured market view.
This is the "eyes" of the system — it reads the chart and makes a call.
"""

import logging
from agents import base

log = logging.getLogger("TechnicalAnalyst")

SYSTEM = """You are a senior forex and commodities technical analyst with 20 years
of institutional order flow experience. You read indicators with precision, identify
high-probability setups, and give clear, decisive assessments. You make a call and
stand behind it with specific evidence.

REGIME-SPECIFIC ANALYSIS:
  TRENDING_UP / TRENDING_DOWN (ADX > 20):
    → Look for: EMA alignment, momentum confirmation, breakout from consolidation.
    → Bias_strength 7-10 = strong trend setup.
    → Report the direction of the trend as your primary bias.

  RANGING (ADX < 20, oscillating):
    → Switch to MEAN-REVERSION mode. Trend indicators are UNRELIABLE here.
    → BUY setups: RSI < 30 (oversold) + price touching/below lower BB + key support nearby.
    → SELL setups: RSI > 70 (overbought) + price touching/above upper BB + key resistance nearby.
    → Bias_strength 6-8 for clear RSI extreme at BB band (this IS a valid edge).
    → Do NOT say NEUTRAL just because there is no trend. Report the mean-reversion signal.
    → Named setup examples: "BB Lower + RSI Oversold", "Overbought Reversal at Resistance".

  VOLATILE:
    → Note the elevated volatility. Look for confirmed direction before committing."""


def analyse(ctx: dict, cfg) -> dict:
    """
    Analyse the current market context and produce a structured bull view.
    Returns a structured analysis dict with bias, strength, support/resistance, etc.
    """
    candle_text = "\n".join(
        f"  {c['time']}: O={c['open']} H={c['high']} L={c['low']} C={c['close']} ({c['direction']})"
        for c in ctx.get("recent_candles", [])
    )

    patterns    = ctx.get("candlestick_patterns", [])
    vol_data    = ctx.get("volume_analysis", {})
    divergence  = ctx.get("rsi_divergence", "NONE")
    htf_bias    = ctx.get("htf_bias", "NEUTRAL")
    regime      = ctx.get("regime", {})

    prompt = f"""
Analyse {ctx['symbol']} on the {ctx['timeframe']} timeframe. Build the bull case.

═══ TREND & EMAs ═══
Price: {ctx['price']}
EMA {cfg.EMA_FAST}: {ctx['ema_fast']} | EMA {cfg.EMA_SLOW}: {ctx['ema_slow']} | EMA {cfg.EMA_TREND}: {ctx['ema_trend']}
Fast above slow: {ctx['fast_above_slow']} | Above trend EMA: {ctx['price_above_trend']}
EMA cross UP this bar: {ctx['ema_cross_up']} | DOWN: {ctx['ema_cross_down']}
Higher TF ({cfg.HTF_TIMEFRAME}) bias: {htf_bias}

═══ MOMENTUM ═══
RSI({cfg.RSI_PERIOD}): {ctx['rsi']} [{ctx['rsi_state']}] | Rising: {ctx['rsi_rising']}
RSI divergence: {divergence}
MACD histogram: {ctx['macd_histogram']} | Positive: {ctx['macd_positive']}
MACD cross UP: {ctx['macd_cross_up']} | DOWN: {ctx['macd_cross_down']}

═══ VOLATILITY & BANDS ═══
BB: Upper={ctx['bb_upper']} Mid={ctx['bb_middle']} Lower={ctx['bb_lower']}
Price in bands (0=lower, 1=upper): {ctx['bb_pct']}
BB squeeze (low vol, expect breakout): {ctx['bb_squeeze']}
ATR({cfg.ATR_PERIOD}): {ctx['atr']} | ATR percentile vs last 50 bars: {ctx['atr_percentile']}%

═══ STRUCTURE & CONTEXT ═══
Market structure: {ctx['market_structure']}
Active sessions: {', '.join(ctx.get('sessions', []))}
Regime: {regime.get('regime', 'N/A')} | ADX: {regime.get('adx', 'N/A')} | Hurst: {regime.get('hurst', 'N/A')}

═══ CANDLESTICK PATTERNS ═══
Detected patterns: {patterns if patterns else 'None'}

═══ VOLUME ═══
Relative volume: {vol_data.get('rel_volume', 1.0):.1f}x average | Signal: {vol_data.get('signal', 'N/A')}
Volume climax: {vol_data.get('climax', False)}

═══ RECENT CANDLES ═══
{candle_text}

CRITICAL — Check the REGIME first:
  • If RANGING: apply mean-reversion logic. RSI<30 at lower BB = BUY setup (bias_strength 6-8).
    RSI>70 at upper BB = SELL setup (bias_strength 6-8). Do NOT default to NEUTRAL.
  • If TRENDING: apply momentum/trend logic. Confirm EMA direction + ADX.

Produce your technical analysis. Be specific and decisive. Name the setup.

Respond in this exact JSON format:
{{
  "trend": "BULLISH" | "BEARISH" | "SIDEWAYS",
  "trend_strength": 1-10,
  "momentum": "STRONG_UP" | "UP" | "NEUTRAL" | "DOWN" | "STRONG_DOWN",
  "bias": "BUY" | "SELL" | "NEUTRAL",
  "bias_strength": 1-10,
  "key_support": <price as number>,
  "key_resistance": <price as number>,
  "setup_name": "name of the specific pattern or null",
  "risks": ["specific technical risk 1", "specific technical risk 2"],
  "summary": "2-3 sentences. What the chart says, why, and the specific entry trigger."
}}
"""

    result = base.call_json(prompt, SYSTEM, cfg, agent_name="technical_analyst")
    if not result:
        log.warning("Technical analyst returned empty response.")
        return {
            "bias": "NEUTRAL", "bias_strength": 0,
            "trend": "SIDEWAYS", "summary": "Analysis unavailable.",
            "risks": [], "key_support": 0, "key_resistance": 0,
        }

    log.info(
        f"Bull view: {result.get('bias','?')} ({result.get('bias_strength','?')}/10) | "
        f"Setup: {result.get('setup_name','none')} | "
        f"{result.get('summary','')[:80]}"
    )
    return result
