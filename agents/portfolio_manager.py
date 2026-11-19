"""
Fund Manager Agent — the final arbitrator of the bull/bear debate.

Receives the full intelligence package:
  - Technical Analyst (bull case, indicators)
  - Bear Analyst (adversarial challenge)
  - News/Macro Analyst (macro backdrop)
  - Past memories (what worked in similar situations)

Debates both sides and makes the final BUY/SELL/HOLD decision.
This is the most important agent — quality of reasoning here determines profitability.
"""

import logging
from agents import base

log = logging.getLogger("FundManager")

SYSTEM = """You are the Chief Investment Officer (CIO) of a professional multi-strategy trading desk.
You receive a structured bull/bear debate across all available signals and make the final decision.

Your mandate is CALIBRATED TRADING — not maximum caution. You already have five independent
filters protecting you: bear analyst veto, risk manager veto, meta-label ML gate, risk governor,
and daily loss limits. These layers will catch bad trades AFTER you decide. Your job is to
identify REAL EDGES and ACT on them with appropriate conviction — not to be a sixth conservative
layer on top of everything else.

CONFIDENCE SCALE (calibrated):
  80-100%: Exceptional setup — strong confluence, clear macro alignment, historical memory confirms
  65-79%:  Good setup — multiple signals agree, reasonable risk-reward, proceed confidently
  55-64%:  Solid edge — more signals for than against; take the trade
  50-54%:  Marginal edge — slight lean, but worth taking with reduced size
  < 50%:   Insufficient edge — HOLD and wait for better opportunity

REGIME-SPECIFIC TRADING STRATEGY:
  TRENDING_UP / TRENDING_DOWN:
    → Trend-following logic. Confirm EMA alignment, ADX > 20, higher-TF bias alignment.
    → Strong momentum + macro alignment = 70-85% confidence.
    → Counter-trend reversals need overwhelming evidence.

  RANGING (ADX < 25, low-momentum oscillation):
    → Mean-REVERSION logic. Do NOT apply trend-following expectations here.
    → BUY signals: RSI < 30 (oversold) + price at key support + lower BB band touch.
    → SELL signals: RSI > 70 (overbought) + price at key resistance + upper BB band touch.
    → Clear RSI extreme at BB band = 65-75% confidence — this IS a high-quality trade.
    → Do NOT penalise confidence because there is no trend; range-bound extremes are the signal.
    → Bias_strength from technical analyst may be LOW in RANGING — this is normal, ignore it.

  VOLATILE:
    → Reduce size but don't always HOLD. Volatility = opportunity if direction is clear.
    → Require TWO additional confirmations before trading.

Capital preservation matters, but so does executing when the edge is real."""


def _build_debate_transcript(
    technical_analysis: dict,
    bear_analysis: dict,
    news_analysis: dict,
    agent_accuracy: dict | None = None,
    extra_signals: dict | None = None,
) -> str:
    """
    Format a structured debate transcript from bull/bear arguments.
    This gives the Fund Manager a narrative to arbitrate rather than raw JSON.
    """
    extra = extra_signals or {}
    acc   = agent_accuracy or {}

    # Accuracy context for Fund Manager to weight agents appropriately
    acc_note = ""
    if acc.get("n_trades", 0) >= 10:
        acc_note = (
            f"\n[AGENT TRACK RECORD — last {acc['n_trades']} trades] "
            f"Technical Analyst accuracy: {acc['tech_accuracy']:.0%} | "
            f"Bear Analyst accuracy (high-risk calls): {acc['bear_accuracy']:.0%} | "
            f"Fund Manager high-confidence accuracy: {acc['pm_accuracy']:.0%}"
        )

    # Extra signals line (COT, economic surprise, Fibonacci, SR zones)
    signals_line = ""
    cot_sig  = extra.get("cot_signal", "")
    econ_sig = extra.get("econ_surprise_signal", "")
    fib_lbl  = extra.get("fib_label", "")
    sr_type  = extra.get("sr_zone_type", "")
    sr_str   = extra.get("sr_zone_strength", 0.0)

    parts = []
    if cot_sig and cot_sig != "NEUTRAL":
        parts.append(f"COT: {cot_sig} (institutional positioning)")
    if econ_sig and econ_sig != "NEUTRAL":
        parts.append(f"Economic Surprise: {econ_sig}")
    if fib_lbl:
        parts.append(f"Fibonacci: price near {fib_lbl} retracement")
    if sr_type:
        parts.append(f"S/R Zone: {sr_type} ({sr_str:.0%} strength)")
    if parts:
        signals_line = "\n\n═══ ADDITIONAL MARKET SIGNALS ═══\n" + " | ".join(parts)

    bull_points = [
        f"• {technical_analysis.get('trend', 'N/A')} trend with {technical_analysis.get('momentum', 'N/A')} momentum",
        f"• Bias strength: {technical_analysis.get('bias_strength', '?')}/10",
        f"• {technical_analysis.get('summary', '')[:100]}",
    ]

    bear_points = [
        f"• {bear_analysis.get('bear_thesis', '')[:100]}",
        f"• Counter to bull: {bear_analysis.get('counter_to_bull', 'None')[:80]}",
        f"• Regime warning: {bear_analysis.get('regime_warning', 'None')}",
    ]

    return f"""
═══ STRUCTURED DEBATE TRANSCRIPT ═══{acc_note}

ROUND 1 — BULL CASE (Technical Analyst, {technical_analysis.get('bias')} bias):
{chr(10).join(bull_points)}
Macro support: {news_analysis.get('macro_tone', 'NEUTRAL')} tone | Session: {news_analysis.get('session_bias', 'N/A')}

ROUND 2 — BEAR CHALLENGE (Adversarial Analyst, risk score {bear_analysis.get('risk_score', 5)}/10):
{chr(10).join(bear_points)}
Evidence cited: {bear_analysis.get('evidence', [])}

ROUND 3 — DISPUTE: {_identify_key_dispute(technical_analysis, bear_analysis)}{signals_line}
"""


def _identify_key_dispute(tech: dict, bear: dict) -> str:
    """Identify the single most important point of disagreement."""
    if bear.get("regime_warning"):
        return f"Regime disagreement — Bear warns: {bear['regime_warning'][:80]}"
    if bear.get("counter_to_bull"):
        return f"Core bull/bear conflict — {bear['counter_to_bull'][:80]}"
    return f"Bull ({tech.get('bias_strength', 5)}/10 strength) vs Bear risk ({bear.get('risk_score', 5)}/10)"


def decide(
    technical_analysis: dict,
    bear_analysis: dict,
    news_analysis: dict,
    past_memories: list,
    ctx: dict,
    regime: dict,
    cfg,
    agent_accuracy: dict | None = None,
    extra_signals: dict | None = None,
) -> dict:
    """
    Make the final trading decision after hearing all sides.

    Returns dict with: action, confidence, reasoning, key_risks, lessons_applied
    """
    if past_memories:
        mem_text = "\n".join(
            f"  [{i+1}] {m['direction']} → {m['outcome']} ({m['pnl_pct']:+.2f}%) | "
            f"Score: {m.get('usefulness_score', 0):.2f} | Lesson: {m['lesson']}"
            for i, m in enumerate(past_memories)
        )
    else:
        mem_text = "  No relevant memories — operating without historical context."

    bear_risk   = bear_analysis.get("risk_score", 5)
    macro_tone  = news_analysis.get("macro_tone", "NEUTRAL")
    regime_name = regime.get("regime", "UNKNOWN")

    debate_transcript = _build_debate_transcript(
        technical_analysis, bear_analysis, news_analysis,
        agent_accuracy=agent_accuracy,
        extra_signals=extra_signals,
    )

    prompt = f"""
You are the final decision-maker on a potential {ctx.get('symbol')} trade.
{debate_transcript}
═══ FULL BULL ANALYST REPORT ═══
Bias: {technical_analysis.get('bias')} (strength {technical_analysis.get('bias_strength')}/10)
Summary: {technical_analysis.get('summary', '')}
Key support: {technical_analysis.get('key_support', 'N/A')} | Key resistance: {technical_analysis.get('key_resistance', 'N/A')}
Trend: {technical_analysis.get('trend', 'N/A')} | Momentum: {technical_analysis.get('momentum', 'N/A')}
Risks: {technical_analysis.get('risks', [])}

═══ FULL BEAR ANALYST REPORT (risk score: {bear_risk}/10) ═══
Bear thesis: {bear_analysis.get('bear_thesis', '')}
Evidence: {bear_analysis.get('evidence', [])}
Challenge to bulls: {bear_analysis.get('counter_to_bull', '')}
Regime warning: {bear_analysis.get('regime_warning', 'None')}

═══ MACRO/NEWS BACKDROP ═══
Macro tone: {macro_tone} | News risk: {news_analysis.get('news_risk', '?')}
Key themes: {news_analysis.get('key_themes', [])}
Session note: {news_analysis.get('session_bias', '')}

═══ MARKET REGIME ═══
Regime: {regime_name} | ADX: {regime.get('adx', '?')} | Hurst: {regime.get('hurst', '?')}
Description: {regime.get('description', '')}

═══ CURRENT MARKET STATE ═══
Price: {ctx.get('price')} | Structure: {ctx.get('market_structure')}
RSI: {ctx.get('rsi')} [{ctx.get('rsi_state')}] | ATR percentile: {ctx.get('atr_percentile')}%
EMA cross UP: {ctx.get('ema_cross_up')} | DOWN: {ctx.get('ema_cross_down')}
BB position: {ctx.get('bb_pct')} | BB squeeze: {ctx.get('bb_squeeze')}
Higher TF bias: {ctx.get('htf_bias', 'N/A')}
Candlestick patterns: {ctx.get('candlestick_patterns', 'N/A')}

═══ MEMORY (similar past situations) ═══
{mem_text}

═══ YOUR TASK ═══
STEP 1 — REGIME CHECK: Is the regime RANGING, TRENDING, or VOLATILE?
  • RANGING → apply mean-reversion logic (RSI extremes, BB touches, S/R levels)
  • TRENDING → apply momentum logic (EMA alignment, ADX, breakout confirmation)
  • VOLATILE → require extra confirmation; apply standard logic but note elevated risk

STEP 2 — BULL vs BEAR WEIGHTING:
  • In RANGING: technical analyst bias_strength is often LOW — this is expected and OK.
    Weight RSI, BB position, and key support/resistance MORE than bias_strength.
  • In TRENDING: weight EMA alignment, ADX, and trend confluences heavily.
  • Score the debate: who made the stronger argument given the regime?

STEP 3 — MACRO ALIGNMENT:
  • Does macro tone SUPPORT or CONTRADICT the trade direction?
  • If macro contradicts: reduce confidence by 10-15 points.
  • If macro supports: add 5-10 points.

STEP 4 — MEMORY APPLICATION:
  • Similar past setups that FAILED: reduce confidence by 10 points each.
  • Similar past setups that WON: add 5 points per lesson.

STEP 5 — FINAL CONFIDENCE (calibrated, not conservative):
  • Use the CONFIDENCE SCALE from your system instructions.
  • In RANGING markets: a clear RSI extreme + BB band touch = start at 65-70%.
  • Bear risk ≥ 8: HOLD unless bull case is overwhelmingly clear (would need 80%+ raw score).
  • Be HONEST about your confidence. The risk system will catch bad trades — your job is finding edges.

IMPORTANT: Write your full chain_of_thought FIRST before deciding on action/confidence.
Reasoning written before the decision is higher quality and less prone to rationalization.

Respond in this exact JSON format:
{{
  "chain_of_thought": "Work through ALL 5 steps here in detail. Cite specific indicator values, memory lessons, and the exact mechanism of your decision. Minimum 3 sentences. This runs BEFORE you commit to action/confidence.",
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0-100,
  "reasoning": "2-3 sentence summary of chain_of_thought for logging",
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "lessons_applied": ["which memory influenced your decision, or 'none'"],
  "bull_bear_verdict": "BULL_WINS" | "BEAR_WINS" | "DRAW"
}}
"""

    result = base.call_json(prompt, SYSTEM, cfg, agent_name="fund_manager")
    if not result or "action" not in result:
        log.warning("Fund manager returned invalid response. Defaulting to HOLD.")
        return {
            "action": "HOLD", "confidence": 0,
            "reasoning": "Agent error — holding for safety.",
            "key_risks": [], "lessons_applied": [], "bull_bear_verdict": "DRAW",
        }

    action     = result.get("action", "HOLD").upper()
    confidence = round(result.get("confidence", 0))

    if action != "HOLD" and confidence < cfg.MIN_CONFIDENCE_TO_TRADE:
        result["action"] = "HOLD"
        result["reasoning"] = (
            f"Confidence {confidence}% below threshold {cfg.MIN_CONFIDENCE_TO_TRADE}%. "
            + result.get("reasoning", "")
        )

    log.info(
        f"Fund Manager: {result.get('action')} ({confidence}%) | "
        f"Verdict: {result.get('bull_bear_verdict','?')} | "
        f"{result.get('reasoning','')[:100]}"
    )
    return result
