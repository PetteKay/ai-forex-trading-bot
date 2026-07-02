"""
Risk Manager Agent — position sizing and trade qualification gate.

Reviews the full debate (bull + bear + macro) and determines:
  1. Whether the trade should be taken at all (qualification gate)
  2. How much capital to risk (position sizing, 0-100% of base lot)
  3. Any additional entry conditions to wait for

Implements meta-labeling: primary model gives direction, this agent
decides IF and HOW MUCH to trade (Kelly-like fraction).
"""

import logging
from agents import base

log = logging.getLogger("RiskManager")

SYSTEM = """You are a professional risk manager at a quantitative hedge fund.
You have one job: protect capital. You are ruthlessly disciplined, never
rationalize a bad trade, and would rather miss 10 good trades than take
1 catastrophic loss. You size positions based on edge quality, not hope."""


def assess(
    ctx: dict,
    tech_analysis: dict,
    bear_analysis: dict,
    news_analysis: dict,
    pm_decision: dict,
    regime: dict,
    cfg,
) -> dict:
    """
    Assess risk and determine position sizing.

    Returns:
        trade_qualified: bool — should this trade be taken?
        size_multiplier: float (0.0 - 1.0) — fraction of base LOT_SIZE to use
        max_risk_pct: float — max % of account to risk on this trade
        veto_reason: str | None — reason for veto (if trade_qualified=False)
        wait_for: str | None — specific condition to wait for before entering
        risk_notes: list[str] — risk observations
    """
    bear_risk  = bear_analysis.get("risk_score", 5)
    pm_conf    = pm_decision.get("confidence", 0)
    macro_risk = news_analysis.get("news_risk", "MEDIUM")
    regime_name= regime.get("regime", "UNKNOWN")
    adx        = regime.get("adx", 20)
    hurst      = regime.get("hurst", 0.5)

    prompt = f"""
Review this trade opportunity and determine position sizing and qualification.

═══ MARKET CONTEXT ═══
Symbol: {ctx.get('symbol')} | Price: {ctx.get('price')}
Regime: {regime_name} | ADX: {adx} | Hurst: {hurst}
ATR: {ctx.get('atr')} | ATR percentile: {ctx.get('atr_percentile')}%
BB squeeze: {ctx.get('bb_squeeze')} | Session: {ctx.get('sessions')}

═══ BULL ANALYST ═══
Bias: {tech_analysis.get('bias')} (strength {tech_analysis.get('bias_strength')}/10)
Summary: {tech_analysis.get('summary','')}

═══ BEAR ANALYST (risk score {bear_risk}/10) ═══
Bear thesis: {bear_analysis.get('bear_thesis','')}
Key risks: {bear_analysis.get('evidence', [])}

═══ MACRO ANALYST ═══
Tone: {news_analysis.get('macro_tone','?')} | News risk: {macro_risk}
Key themes: {news_analysis.get('key_themes', [])}

═══ FUND MANAGER DECISION ═══
Action: {pm_decision.get('action','?')} | Confidence: {pm_conf}%
Reasoning: {pm_decision.get('reasoning','')}

═══ YOUR TASK ═══
1. Is this trade qualified? Consider: bear risk score, macro news risk, regime suitability,
   PM confidence, and whether the setup meets a minimum edge threshold.
2. If qualified, what fraction of the base lot should we use? (1.0 = full size, 0.5 = half, etc.)
   High edge (bear < 4, PM conf > 75, regime aligned) → 1.0
   Moderate edge (bear 4-6, PM conf 65-75) → 0.5-0.75
   Low edge (bear > 6 or high macro risk) → VETO or 0.25
3. Is there a specific condition we should wait for before entering?

Respond in JSON:
{{
  "trade_qualified": true | false,
  "size_multiplier": 0.0-1.0,
  "max_risk_pct": 0.5-2.0,
  "veto_reason": "reason for veto or null",
  "wait_for": "specific condition or null",
  "risk_notes": ["note1", "note2"]
}}
"""

    result = base.call_json(prompt, SYSTEM, cfg, agent_name="risk_manager")
    if not result:
        # Conservative defaults on failure
        return {
            "trade_qualified": pm_conf >= cfg.MIN_CONFIDENCE_TO_TRADE,
            "size_multiplier": 0.5,
            "max_risk_pct": 1.0,
            "veto_reason": None,
            "wait_for": None,
            "risk_notes": ["Risk assessment unavailable — using conservative defaults"],
        }

    qualified = bool(result.get("trade_qualified", False))
    size_mult = float(result.get("size_multiplier", 0.5))
    veto      = result.get("veto_reason")

    if qualified:
        log.info(f"Risk Manager: APPROVED | size={size_mult:.0%} | risk_pct={result.get('max_risk_pct','?')}%")
    else:
        log.info(f"Risk Manager: VETOED | {veto}")

    return result
