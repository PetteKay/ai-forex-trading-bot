"""
Reflection Agent — the learning engine.

After every significant trade closes, this agent:
  1. Reviews what the other agents said at entry
  2. Looks at what actually happened
  3. Extracts a specific, actionable lesson
  4. Scores how useful that lesson is (0-1)
  5. Only returns a lesson if it's worth saving (above threshold)

This is how the bot gets smarter with every trade.
The most important design principle: be EXTREMELY selective.
Saving mediocre lessons pollutes the memory and makes future decisions worse.
"""

import logging
from agents import base

log = logging.getLogger("Reflection")

SYSTEM = """You are an elite trading coach who reviews completed trades to extract
pure, distilled wisdom. You are brutally selective — you only save lessons that are
genuinely surprising, novel, or critical to avoid repeating a mistake. You never write
generic advice like "wait for confirmation" or "follow the trend." Every lesson you save
must be specific enough that a trader reading it would immediately know what to do
differently next time."""


def _pre_filter(r_multiple: float, pm_confidence: int, outcome: str) -> bool:
    """
    Quick check: is this trade worth spending an LLM call on?
    Uses R-multiple (profit / initial risk) so the filter is account-size-invariant.
    A 1.0 R-multiple means we made back exactly what we risked.
    """
    # Always reflect on stop-losses — the primary source of learning
    if outcome in ("STOP_LOSS", "STOP_LOSS_FULL"):
        return True
    # High-confidence agent call that was wrong = critical lesson
    if pm_confidence >= 80 and r_multiple < 0:
        return True
    # Exceptional win (1.5R = made 150% of risk back)
    if r_multiple >= 1.5:
        return True
    # Any meaningful outcome — 0.3R threshold catches 3-pip winners on a 10-pip SL
    if abs(r_multiple) >= 0.3:
        return True
    return False


def reflect(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    r_multiple: float,
    hold_hours: float,
    outcome: str,
    entry_context: dict,
    technical_analysis: dict,
    pm_decision: dict,
    cfg,
) -> dict | None:
    """
    Review a completed trade and extract a lesson if it's worth saving.
    r_multiple: net_pnl / initial_risk_usd — account-size-invariant outcome measure.
    """
    pm_confidence = int(pm_decision.get("confidence", 0))

    # Pre-filter using R-multiple (not account % — account-size invariant)
    if not _pre_filter(r_multiple, pm_confidence, outcome):
        log.debug(f"Trade skipped for reflection (R={r_multiple:+.2f}, confidence={pm_confidence}%, outcome={outcome})")
        return None

    prompt = f"""
Review this completed trade and extract a lesson — IF AND ONLY IF there is one worth saving.

═══ TRADE RESULT ═══
Symbol: {symbol} | Direction: {direction}
Entry: {entry_price:.5f} | Exit: {exit_price:.5f}
P&L: {pnl_pct:+.2f}% (R={r_multiple:+.2f}) | Outcome: {outcome}
Hold time: {hold_hours:.1f} hours

═══ MARKET STATE AT ENTRY ═══
Structure: {entry_context.get('market_structure')} | Sessions: {entry_context.get('sessions')}
RSI: {entry_context.get('rsi')} [{entry_context.get('rsi_state')}]
EMA cross up: {entry_context.get('ema_cross_up')} | EMA cross down: {entry_context.get('ema_cross_down')}
Fast above slow: {entry_context.get('fast_above_slow')} | Above trend: {entry_context.get('price_above_trend')}
MACD positive: {entry_context.get('macd_positive')} | BB position: {entry_context.get('bb_pct')}
ATR percentile (volatility): {entry_context.get('atr_percentile')}%
Volatility squeeze: {entry_context.get('bb_squeeze')}

═══ WHAT THE TECHNICAL ANALYST SAID ═══
Bias: {technical_analysis.get('bias')} (strength {technical_analysis.get('bias_strength')}/10)
Summary: {technical_analysis.get('summary', 'N/A')}
Risks flagged: {technical_analysis.get('risks', [])}

═══ WHAT THE PORTFOLIO MANAGER DECIDED ═══
Action: {pm_decision.get('action')} | Confidence: {pm_decision.get('confidence')}%
Reasoning: {pm_decision.get('reasoning', 'N/A')}
Risks flagged: {pm_decision.get('key_risks', [])}

═══ YOUR TASK ═══
Extract a specific, reusable lesson from this trade.

USEFULNESS SCORING GUIDE:
  0.8-1.0: Exceptional — counterintuitive result, critical failure lesson, or a pattern
            that clearly works/fails under specific conditions. SAVE.
  0.6-0.8: Clear lesson — specific enough that future behavior should change. SAVE.
  0.4-0.6: Mild insight — borderline. Only save if genuinely new information.
  0.0-0.5: Not worth saving — the result was expected, generic, or offers no new guidance.

Respond in this exact JSON format:
{{
  "should_save": true | false,
  "usefulness_score": 0.0-1.0,
  "lesson": "Specific, actionable insight (required if should_save=true, else empty string)",
  "pattern_name": "Short 2-4 word name for this setup, or null",
  "category": "ENTRY_TIMING" | "EXIT_TIMING" | "REGIME_FILTER" | "RISK_MANAGEMENT" | "VOLATILITY" | "SESSION" | null
}}

Be ruthlessly selective. Most trades teach nothing new.
"""

    result = base.call_json(prompt, SYSTEM, cfg, agent_name="reflection")
    if not result or not result.get("should_save"):
        log.debug("Reflection: nothing worth saving.")
        return None

    lesson = result.get("lesson", "").strip()
    if not lesson or len(lesson) < 15:
        return None

    log.info(f"Reflection saved (score={result.get('usefulness_score', 0):.2f}): {lesson[:100]}")

    # Assemble the full memory record
    return {
        "symbol":           symbol,
        "direction":        direction,
        "entry_price":      entry_price,
        "exit_price":       exit_price,
        "pnl_pct":          pnl_pct,
        "outcome":          outcome,
        "fingerprint":      _fingerprint(entry_context),
        "market_context":   {k: v for k, v in entry_context.items() if k != "_df"},
        "agent_reasoning":  f"Tech: {technical_analysis.get('summary','')} | PM: {pm_decision.get('reasoning','')}",
        "lesson":           lesson,
        "usefulness_score": float(result.get("usefulness_score", 0)),
        "pattern_name":     result.get("pattern_name"),
        "category":         result.get("category"),
    }


def _fingerprint(ctx: dict) -> str:
    from strategy.indicators import market_fingerprint
    return market_fingerprint(ctx)
