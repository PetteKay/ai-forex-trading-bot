"""
News & Macro Analyst Agent.

Assesses the macro/news backdrop for the current trading session.
Reads manually-provided news context or uses the economic calendar
to infer expected macro tone.

Without a real-time news feed, this agent reasons from:
  - The economic calendar (what events happened/are upcoming)
  - Session context (London = EU-driven, NY = USD-driven)
  - Symbol characteristics (EURUSD is ECB/Fed-sensitive, XAUUSD is risk/inflation)
  - User-supplied news snippets (optional)
"""

import logging
from agents import base

log = logging.getLogger("NewsAnalyst")

try:
    from data.macro_context import get_macro_context, macro_summary_for_llm
    _MACRO_AVAILABLE = True
except ImportError:
    _MACRO_AVAILABLE = False

SYSTEM = """You are a macro strategist and market analyst with deep expertise in
how central bank policy, economic data releases, and geopolitical events affect
forex and commodity prices. You think in terms of risk-on/risk-off, dollar strength
cycles, yield differentials, and sentiment regimes."""


# Symbol-to-macro-drivers mapping — covers all 30 watch-list instruments
SYMBOL_DRIVERS = {
    # ── Forex Majors ─────────────────────────────────────────────────────────
    "EURUSD": "ECB vs Fed policy divergence, EU inflation, US CPI/NFP, PMIs, EUR-area growth",
    "GBPUSD": "BoE vs Fed divergence, UK inflation, UK employment, post-Brexit trade flows",
    "USDJPY": "BoJ yield curve control vs US yields, carry trade flows, risk-on/off sentiment",
    "AUDUSD": "RBA policy, China growth data, iron ore / commodity prices, risk appetite",
    "NZDUSD": "RBNZ policy, NZ dairy/commodity exports, China demand, risk sentiment",
    "USDCAD": "BoC policy vs Fed, oil prices (WTI), Canada employment, trade balance",
    "USDCHF": "SNB intervention history, safe-haven flows, EU risk contagion, risk-off",

    # ── JPY Crosses ───────────────────────────────────────────────────────────
    "GBPJPY": "BoE vs BoJ divergence, UK data + Japanese risk-off flows, dual volatility source",
    "EURJPY": "ECB vs BoJ policy gap, eurozone PMIs, BoJ yield curve control, carry unwind risk",
    "AUDJPY": "RBA vs BoJ, China growth proxy, risk-on = AUD/JPY up, risk-off = sharp unwind",
    "NZDJPY": "RBNZ vs BoJ, NZ commodity exports, risk appetite gauge, carry trade vehicle",
    "CADJPY": "BoC vs BoJ, WTI oil prices (CAD), risk sentiment (JPY), North American data",
    "CHFJPY": "SNB safe-haven vs BoJ ultra-loose policy, EU crisis risk, dual safe-haven conflict",

    # ── GBP Crosses ───────────────────────────────────────────────────────────
    "GBPNZD": "BoE vs RBNZ policy divergence, UK data vs NZ commodity cycle, most volatile pair",
    "GBPAUD": "BoE vs RBA divergence, UK economic outlook vs Australian commodities/China demand",
    "GBPCAD": "BoE vs BoC divergence, UK data + Canadian oil prices, triple macro driver",
    "GBPCHF": "BoE vs SNB divergence, UK risk events vs Swiss safe-haven demand",

    # ── EUR/NZD Cross ─────────────────────────────────────────────────────────
    "EURNZD": "ECB vs RBNZ policy gap, EU growth vs NZ commodity cycle, dairy prices",

    # ── Metals ────────────────────────────────────────────────────────────────
    "XAUUSD": "Real yields (US10Y minus inflation), USD strength/weakness, geopolitical risk, central bank gold buying, recession fear",
    "XAGUSD": "Gold correlation (safe-haven), industrial demand (solar, EVs), USD, mining supply",

    # ── Energies ──────────────────────────────────────────────────────────────
    "USOIL":  "OPEC+ production decisions, US EIA inventory data (Wed 14:30 UTC), geopolitical supply risk, USD strength",
    "UKOIL":  "Brent crude: OPEC+ decisions, North Sea production, EU energy policy, geopolitical risk",
    "NATGAS": "US storage reports (Thu 14:30 UTC), seasonal demand (heating/cooling), LNG exports, weather forecasts",

    # ── Equity Indices ────────────────────────────────────────────────────────
    "NAS100": "Fed rate path, mega-cap tech earnings (AAPL MSFT NVDA GOOGL), AI investment cycle, growth vs value rotation",
    "US30":   "Fed policy, blue-chip earnings, US economic health, consumer sentiment, industrial output",
    "SPX500": "Fed policy + earnings cycle, VIX regime, credit spreads, sector rotation",
    "GER40":  "ECB policy, German industrial output, EUR/USD rate, eurozone PMIs, global trade",
    "UK100":  "BoE policy, GBP strength, FTSE mining/energy sector (commodity-heavy index), UK GDP",
    "JPN225": "BoJ policy, USD/JPY rate (weak JPY = Nikkei up), Japanese corporate earnings, export data",
    "AUS200": "RBA policy, Australian mining/financial sector, China growth data, AUD/USD rate",

    # ── Crypto CFDs ───────────────────────────────────────────────────────────
    "BTCUSD": "Bitcoin halving cycle, ETF flows, Fed liquidity (risk-on = BTC up), regulatory news, on-chain metrics",
    "ETHUSD": "Ethereum protocol upgrades, ETH staking yields, DeFi activity, BTC correlation, crypto regulatory news",
}


def analyse(
    symbol: str,
    sessions: list,
    upcoming_events: list,
    cfg,
    news_snippets: list = None,
    macro: dict = None,
) -> dict:
    """
    Assess the macro/news backdrop.

    Args:
        symbol: trading instrument
        sessions: currently active sessions (LONDON, NEW_YORK, ASIAN, etc.)
        upcoming_events: list of dicts from EconomicCalendar.next_event()
        cfg: bot config
        news_snippets: optional list of recent news headlines (strings)

    Returns dict with:
        macro_tone: BULLISH | BEARISH | NEUTRAL | UNCERTAIN
        news_risk: LOW | MEDIUM | HIGH
        key_themes: list[str]
        session_bias: str
        recommendation: str
    """
    drivers   = SYMBOL_DRIVERS.get(symbol.upper(), f"macro factors for {symbol}")
    session_s = ", ".join(sessions) if sessions else "off-hours"
    events_s  = "\n".join([
        f"  - {e['name']} in {int((e['time'] - __import__('datetime').datetime.now(__import__('datetime').timezone.utc)).total_seconds() / 60)}min"
        for e in (upcoming_events or [])
    ]) or "  None scheduled imminently"

    snippets_s = ""
    if news_snippets:
        snippets_s = "\nRecent headlines:\n" + "\n".join(f"  • {h}" for h in news_snippets[:6])

    # Live macro signals (VIX, DXY, yields, gold, fear & greed)
    live_macro = macro or ({} if not _MACRO_AVAILABLE else get_macro_context(symbol))
    macro_s = ""
    if live_macro and _MACRO_AVAILABLE:
        macro_s = "\nLive macro signals:\n" + macro_summary_for_llm(live_macro, symbol)

    prompt = f"""
Assess the macro/news environment for a potential trade.

Symbol: {symbol}
Key macro drivers: {drivers}
Active sessions: {session_s}
Upcoming high-impact events:
{events_s}{snippets_s}{macro_s}

Using the live macro signals above (VIX regime, DXY trend, yield direction, fear/greed score)
AND typical session behaviour AND the economic calendar events:

1. What is the macro tone for {symbol} right now?
2. Is there elevated news risk that should reduce position size or suggest waiting?
3. What are the 2-3 key themes driving this instrument today?
4. Given the session and LIVE macro signals, does the backdrop favour bulls or bears?
5. Does the VIX/DXY reading support or contradict a trade right now?

Respond in JSON:
{{
  "macro_tone": "BULLISH" | "BEARISH" | "NEUTRAL" | "UNCERTAIN",
  "news_risk": "LOW" | "MEDIUM" | "HIGH",
  "key_themes": ["theme1", "theme2"],
  "session_bias": "Brief statement about how this session typically moves {symbol}",
  "recommendation": "What should the trader do given this macro backdrop (1-2 sentences)",
  "vix_impact": "How current VIX level affects this trade",
  "dxy_impact": "How current DXY trend affects {symbol}"
}}
"""

    result = base.call_json(prompt, SYSTEM, cfg, agent_name="news_analyst")
    if not result:
        return {
            "macro_tone":   "NEUTRAL",
            "news_risk":    "MEDIUM",
            "key_themes":   [],
            "session_bias": "Unknown",
            "recommendation": "Proceed with caution — macro analysis unavailable.",
        }

    log.info(f"Macro: {result.get('macro_tone','?')} | News risk: {result.get('news_risk','?')} | {result.get('recommendation','')[:80]}")
    return result
