"""
Agent desk runner — the LLM part of one symbol's evaluation.

Contains no MetaTrader 5 calls, so desks for several symbols can run on worker
threads while every MT5 call stays on the main thread (the MT5 Python package
is not documented as thread-safe).

Inside one desk:
    Technical ──► Bear ─────┐
    Headlines + Macro ► News ┴──► Fund Manager
The News analyst does not use the Technical analyst's output, so it now starts
at the same time instead of after it (it used to wait for Technical, then for
the headline and macro fetches).
"""

import logging
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("Desk")

_BEAR_DEFAULT = {"bear_thesis": "Analysis unavailable.", "risk_score": 5,
                 "evidence": [], "counter_to_bull": "Unknown", "regime_warning": ""}
_NEWS_DEFAULT = {"macro_tone": "NEUTRAL", "news_risk": 5, "key_themes": [],
                 "session_bias": "NEUTRAL", "recommendation": "HOLD"}


def run_desk(job: dict, cfg, agents: dict, fetch_headlines, get_macro_context=None) -> dict:
    """
    job:    per-symbol inputs prepared on the main thread (symbol, ctx, regime_info,
            memories, upcoming_events, agent_accuracy, extra_signals).
    agents: {"technical", "bear", "news", "portfolio_manager"} modules.
    Returns {"tech", "bear", "macro", "pm", "live_headlines", "live_macro"}.
    """
    symbol = job["symbol"]
    ctx    = job["ctx"]

    def _news_side():
        headlines = []
        try:
            headlines = fetch_headlines(symbol) or []
        except Exception as e:
            log.debug(f"[{symbol}] Headline fetch failed: {e}")
        if headlines:
            log.info(f"[{symbol}]   Live headlines: {len(headlines)} | Top: {headlines[0][:80]}")
        live_macro = {}
        if get_macro_context is not None:
            try:
                live_macro = get_macro_context(symbol) or {}
            except Exception as e:
                log.debug(f"[{symbol}] Macro fetch skipped: {e}")
        try:
            news = agents["news"].analyse(
                symbol          = symbol,
                sessions        = ctx.get("sessions", []),
                upcoming_events = job.get("upcoming_events", []),
                cfg             = cfg,
                news_snippets   = headlines or None,
                macro           = live_macro or None,
            )
        except Exception as e:
            log.warning(f"[{symbol}] News analyst failed: {e} — using safe default")
            news = dict(_NEWS_DEFAULT)
        return news, headlines, live_macro

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"news-{symbol}") as pool:
        news_fut = pool.submit(_news_side)

        log.info(f"[{symbol}] [1/5] Technical Analyst (News analyst running in parallel)...")
        tech = agents["technical"].analyse(ctx, cfg)
        log.info(f"[{symbol}]   Bull: {tech.get('bias','?')} ({tech.get('bias_strength','?')}/10) | "
                 f"{str(tech.get('summary',''))[:80]}")

        log.info(f"[{symbol}] [2/5] Bear Analyst...")
        try:
            bear = agents["bear"].analyse(ctx, tech, cfg)
        except Exception as e:
            log.warning(f"[{symbol}] Bear analyst failed: {e} — using safe default")
            bear = dict(_BEAR_DEFAULT)

        macro, live_headlines, live_macro = news_fut.result()

    log.info(f"[{symbol}]   Bear risk: {bear.get('risk_score','?')}/10 | "
             f"{str(bear.get('bear_thesis',''))[:80]}")
    log.info(f"[{symbol}]   Macro: {macro.get('macro_tone','?')} | News risk: {macro.get('news_risk','?')}")

    log.info(f"[{symbol}] [4/5] Fund Manager (arbitrates bull vs bear)...")
    pm = agents["portfolio_manager"].decide(
        technical_analysis = tech,
        bear_analysis      = bear,
        news_analysis      = macro,
        past_memories      = job["memories"],
        ctx                = ctx,
        regime             = job["regime_info"],
        cfg                = cfg,
        agent_accuracy     = job.get("agent_accuracy"),
        extra_signals      = job["extra_signals"],
    )
    return {"tech": tech, "bear": bear, "macro": macro, "pm": pm,
            "live_headlines": live_headlines, "live_macro": live_macro}
