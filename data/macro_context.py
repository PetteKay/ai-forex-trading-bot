"""
Real-time Macro Context Fetcher.

Fetches live macro signals via yfinance (free, no API key needed):
  - VIX       → equity fear / risk-off proxy
  - DXY       → US Dollar strength (critical for all forex pairs)
  - US10Y     → Treasury yield (rate environment)
  - Gold      → safe-haven demand
  - TLT       → bond price proxy (inverse bond volatility)
  - SPX       → equity risk-on/off signal
  - USDJPY    → yen carry trade
  - Fear&Greed→ CNN sentiment index (urllib, no extra deps)

All fetches are non-blocking / best-effort — a failed fetch returns None
and the bot continues without that signal rather than crashing.

Requires: yfinance  (py -m pip install yfinance)
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError

log = logging.getLogger("MacroContext")

_cache: dict = {}
_cache_ts: float = 0.0
_fetch_lock = threading.Lock()
# After a failed fetch (empty result), wait this long before trying again
# instead of spawning a new 25 s fetch for every symbol.
_FAIL_BACKOFF_SECONDS = 120
_CACHE_TTL = 900  # 15 minutes — yfinance 1h bars; no need to refresh every 5 min


def _yf_close(ticker_symbol: str) -> tuple[float | None, float | None]:
    """
    Return (latest_close, previous_close) for a ticker.
    Uses individual Ticker.history() to avoid MultiIndex issues from batch download.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker_symbol).history(period="3d", interval="1h", auto_adjust=True)
        col = hist["Close"].dropna()
        latest = float(col.iloc[-1]) if len(col) >= 1 else None
        prev   = float(col.iloc[-2]) if len(col) >= 2 else None
        return latest, prev
    except Exception:
        return None, None


def _fetch_fear_greed() -> tuple[float | None, str]:
    """CNN Fear & Greed Index (0=extreme fear, 100=extreme greed). Returns (score, label)."""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ForexBot/2.0)"})
        with urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
        fg = data.get("fear_and_greed", {})
        score  = round(float(fg["score"]), 1)
        label  = fg.get("rating", "").upper().replace(" ", "_")
        return score, label
    except Exception:
        return None, ""


def _fetch_all(symbol: str) -> dict:
    """
    Do the actual yfinance + Fear&Greed fetch. Runs inside an isolated child
    process (see process_isolation.run_isolated) — must stay a plain,
    module-level, picklable function with no reliance on module globals for
    its result (the cache write happens in the parent, in get_macro_context).
    """
    try:
        import yfinance  # noqa — raise ImportError early if not installed
    except ImportError:
        log.warning("yfinance not installed — macro context unavailable. Run: py -m pip install yfinance")
        return {}

    ctx: dict = {}

    # VIX — equity fear gauge
    vix, vix_prev = _yf_close("^VIX")
    if vix is not None:
        ctx["vix"] = round(vix, 2)
        if vix < 15:
            ctx["vix_regime"] = "CALM"
        elif vix < 20:
            ctx["vix_regime"] = "NORMAL"
        elif vix < 30:
            ctx["vix_regime"] = "ELEVATED"
        else:
            ctx["vix_regime"] = "FEAR"

    # DXY — US Dollar Index
    dxy, dxy_prev = _yf_close("DX-Y.NYB")
    if dxy is not None:
        ctx["dxy"] = round(dxy, 3)
        if dxy_prev is not None:
            ctx["dxy_change_pct"] = round(((dxy - dxy_prev) / dxy_prev) * 100, 3)
            ctx["dxy_trend"] = "STRENGTHENING" if dxy > dxy_prev else "WEAKENING"

    # US 10Y yield
    us10y, us10y_prev = _yf_close("^TNX")
    if us10y is not None:
        ctx["us10y_yield"] = round(us10y, 3)
        if us10y_prev is not None:
            ctx["yields_direction"] = "RISING" if us10y > us10y_prev else "FALLING"

    # Gold — safe-haven / risk-off
    gold, gold_prev = _yf_close("GC=F")
    if gold is not None:
        ctx["gold_price"] = round(gold, 2)
        if gold_prev is not None:
            ctx["gold_trend"] = "RISING" if gold > gold_prev else "FALLING"

    # TLT bond ETF — inverse bond vol proxy
    tlt, tlt_prev = _yf_close("TLT")
    if tlt is not None:
        ctx["tlt_price"] = round(tlt, 2)
        if tlt_prev is not None:
            ctx["bond_sentiment"] = "RISK_OFF" if tlt > tlt_prev else "RISK_ON"

    # S&P 500 — equity risk appetite
    spx, spx_prev = _yf_close("^GSPC")
    if spx is not None:
        ctx["spx"] = round(spx, 2)
        if spx_prev is not None:
            spx_chg = ((spx - spx_prev) / spx_prev) * 100
            ctx["spx_change_pct"] = round(spx_chg, 3)
            ctx["spx_trend"] = "BULLISH" if spx_chg > 0 else "BEARISH"

    # USDJPY — yen carry trade health
    jpy, jpy_prev = _yf_close("USDJPY=X")
    if jpy is not None:
        ctx["usdjpy"] = round(jpy, 3)
        if jpy_prev is not None:
            ctx["carry_trade"] = "ACTIVE" if jpy > jpy_prev else "UNWINDING"

    # CNN Fear & Greed (no requests dep — uses urllib)
    fg_score, fg_label = _fetch_fear_greed()
    if fg_score is not None:
        ctx["fear_greed_score"] = fg_score
        ctx["fear_greed_label"] = fg_label

    ctx["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return ctx


def _fresh(now: float) -> bool:
    if not _cache_ts:
        return False
    ttl = _CACHE_TTL if _cache else _FAIL_BACKOFF_SECONDS
    return (now - _cache_ts) < ttl


def get_macro_context(symbol: str = "EURUSD") -> dict:
    """
    Return a dict of live macro signals relevant to the given symbol.
    Results are cached for 15 minutes.
    Returns {} on total failure — callers must handle empty dict gracefully.

    The actual fetch runs in an isolated child process (see
    data.process_isolation.run_isolated): yfinance's curl_cffi backend has
    documented Windows access-violation crashes on network failures, which
    kill the whole interpreter and can't be caught by a Python try/except.
    Isolating the fetch means that crash can only take down the child —
    the live trading bot keeps running and just gets {} back, same as any
    other best-effort fetch failure.
    """
    global _cache, _cache_ts
    now = time.time()
    if _fresh(now):
        return _cache

    # Agent desks run on several threads; only one spawns the fetch process.
    with _fetch_lock:
        if _fresh(time.time()):   # filled while we waited
            return _cache
        from data.process_isolation import run_isolated
        ctx = run_isolated(_fetch_all, symbol, timeout=25.0, default={})
        _cache    = ctx
        _cache_ts = time.time()

    if ctx:
        log.info(
            f"Macro: VIX={ctx.get('vix','?')} [{ctx.get('vix_regime','?')}] | "
            f"DXY={ctx.get('dxy','?')} [{ctx.get('dxy_trend','?')}] | "
            f"US10Y={ctx.get('us10y_yield','?')}% [{ctx.get('yields_direction','?')}] | "
            f"Gold={ctx.get('gold_price','?')} [{ctx.get('gold_trend','?')}] | "
            f"F&G={ctx.get('fear_greed_score','?')} [{ctx.get('fear_greed_label','')}]"
        )
    else:
        log.warning("MacroContext: all data sources returned None — bot runs without macro signals")

    return ctx


def macro_summary_for_llm(macro: dict, symbol: str) -> str:
    """
    Convert macro context dict into a concise natural-language string
    for injecting into LLM agent prompts.
    """
    if not macro:
        return "Live macro data unavailable."

    lines = []

    # VIX
    if "vix" in macro:
        lines.append(f"VIX: {macro['vix']} ({macro.get('vix_regime','?')} volatility)")

    # DXY — critical for all forex pairs
    if "dxy" in macro:
        chg = macro.get("dxy_change_pct")
        chg_str = f" ({chg:+.3f}%)" if chg is not None else ""
        lines.append(f"DXY (USD Index): {macro['dxy']}{chg_str} — {macro.get('dxy_trend','?')}")

    # Symbol-specific DXY interpretation
    sym = symbol.upper()
    if "dxy_trend" in macro:
        if sym in ("EURUSD", "GBPUSD", "AUDUSD") and macro["dxy_trend"] == "STRENGTHENING":
            lines.append(f"  → Strong USD = bearish pressure on {sym}")
        elif sym in ("EURUSD", "GBPUSD", "AUDUSD") and macro["dxy_trend"] == "WEAKENING":
            lines.append(f"  → Weak USD = bullish tailwind for {sym}")
        elif sym == "USDJPY" and macro["dxy_trend"] == "STRENGTHENING":
            lines.append(f"  → Strong USD = bullish pressure on {sym}")

    # Yields
    if "us10y_yield" in macro:
        lines.append(f"US 10Y Yield: {macro['us10y_yield']}% — {macro.get('yields_direction','?')}")

    # Gold (risk sentiment)
    if "gold_trend" in macro:
        lines.append(f"Gold: {macro.get('gold_price','?')} — {macro['gold_trend']} ({'risk-off bid' if macro['gold_trend']=='RISING' else 'risk appetite'})")

    # Bond sentiment
    if "bond_sentiment" in macro:
        lines.append(f"Bond market: {macro['bond_sentiment'].replace('_',' ')}")

    # SPX
    if "spx_change_pct" in macro:
        lines.append(f"S&P 500: {macro.get('spx','?')} ({macro['spx_change_pct']:+.2f}%) — {macro.get('spx_trend','?')}")

    # JPY carry
    if "carry_trade" in macro:
        lines.append(f"JPY carry trade: {macro['carry_trade'].replace('_',' ')} (USDJPY={macro.get('usdjpy','?')})")

    # Fear & Greed
    if "fear_greed_score" in macro:
        lines.append(f"CNN Fear & Greed: {macro['fear_greed_score']}/100 — {macro.get('fear_greed_label','?').replace('_',' ')}")

    return "\n".join(lines) if lines else "Macro data fetched but no signals extracted."
