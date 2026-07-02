"""
Bear Analyst Agent — adversarial counterpart to the Technical Analyst.

Where the Technical Analyst finds reasons to trade, the Bear Analyst
actively searches for reasons NOT to. This adversarial tension forces
the Fund Manager to justify each trade against the best bear case.

Returns the strongest bear argument the market data can support.
"""

import logging
from agents import base

log = logging.getLogger("BearAnalyst")

SYSTEM = """You are a world-class short-seller and contrarian analyst.
Your only job is to find every reason the current trade setup could FAIL.
You look for hidden divergences, exhaustion signals, structural weaknesses,
false breakouts, and liquidity traps. You are brilliant, cynical, and right
when bulls are complacent. Be specific — no generic warnings.

REGIME-SPECIFIC BEAR ANALYSIS:
  RANGING (ADX < 25): Do NOT penalise for "lack of trend" — RANGING is the intended regime.
    In RANGING, the bear case is: range might BREAK out against the trade direction.
    The real risks are: approaching a major key level from the wrong side, volume spike
    suggesting breakout, approaching session open/close where reversals often fail.
    Do NOT give high risk scores (7-10) simply because there is no ADX trend.

  TRENDING: Risk the bull thesis with trend exhaustion, divergence, or counter-trend signals.
  VOLATILE: Flag position sizing risk; direction uncertainty earns high risk scores.

Risk score calibration:
  0-3: Clean setup, bear case is weak
  4-5: Some legitimate concerns but manageable
  6-7: Real risks present — fund manager should reduce size or wait for confirmation
  8-10: Setup is genuinely dangerous — veto the trade"""


def analyse(ctx: dict, tech_analysis: dict, cfg) -> dict:
    """
    Build the bear case against the current market setup.

    Returns:
        bear_thesis: str — the strongest bear argument
        risk_score: int (0-10) — how dangerous this setup is (10 = avoid)
        evidence: list[str] — specific data points supporting the bear case
        counter_to_bull: str — direct challenge to the technical analyst's bias
        regime_warning: str | None — any regime-specific risk
    """
    bias       = tech_analysis.get("bias", "NEUTRAL")
    strength   = tech_analysis.get("bias_strength", 5)
    summary    = tech_analysis.get("summary", "")

    prompt = f"""
═══ CURRENT MARKET DATA ═══
Symbol: {ctx.get('symbol')} | Timeframe: {ctx.get('timeframe')}
Price: {ctx.get('price')} | Structure: {ctx.get('market_structure')}
Session: {ctx.get('sessions')}

RSI: {ctx.get('rsi')} [{ctx.get('rsi_state')}] | Rising: {ctx.get('rsi_rising')}
EMA fast/slow/trend: {ctx.get('ema_fast')}/{ctx.get('ema_slow')}/{ctx.get('ema_trend')}
Price above trend EMA: {ctx.get('price_above_trend')}
Fast above slow: {ctx.get('fast_above_slow')}
EMA cross UP: {ctx.get('ema_cross_up')} | EMA cross DOWN: {ctx.get('ema_cross_down')}
MACD histogram: {ctx.get('macd_histogram')} | Positive: {ctx.get('macd_positive')}
BB position: {ctx.get('bb_pct')} (0=lower band, 1=upper band)
BB squeeze: {ctx.get('bb_squeeze')}
ATR: {ctx.get('atr')} | ATR percentile: {ctx.get('atr_percentile')}%
Candlestick patterns: {ctx.get('candlestick_patterns', 'N/A')}
Volume analysis: {ctx.get('volume_analysis', 'N/A')}
Higher TF bias: {ctx.get('htf_bias', 'N/A')}
Regime: {ctx.get('regime', {}).get('regime', 'N/A')} | Hurst: {ctx.get('regime', {}).get('hurst', 'N/A')}

═══ THE BULL ANALYST SAID ═══
Bias: {bias} (strength {strength}/10)
Summary: {summary}

═══ YOUR TASK ═══
Argue the BEAR case as strongly as the data allows. Find every flaw in the bull thesis.
Look for: divergences, exhaustion, overbought/oversold extremes, false breakouts,
low-quality EMA crosses, counter-trend moves, liquidity traps, poor R:R at current levels.

Respond in this exact JSON format:
{{
  "bear_thesis": "The single strongest bear argument (2-3 sentences)",
  "risk_score": 0-10,
  "evidence": ["specific data point 1", "specific data point 2", ...],
  "counter_to_bull": "Direct challenge to the bull analyst's specific claim",
  "regime_warning": "Any regime-specific risk, or null if none"
}}

Be brutally honest. A risk_score of 7+ means the bull thesis has serious flaws.
"""

    result = base.call_json(prompt, SYSTEM, cfg, agent_name="bear_analyst")
    if not result:
        return {
            "bear_thesis": "Unable to generate bear analysis",
            "risk_score": 5,
            "evidence": [],
            "counter_to_bull": "",
            "regime_warning": None,
        }

    log.info(f"Bear risk score: {result.get('risk_score', '?')}/10 | {result.get('bear_thesis','')[:80]}")
    return result
