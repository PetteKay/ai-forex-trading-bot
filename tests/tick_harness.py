"""
One-tick harness: runs the REAL bot.run() main loop for exactly one tick with a
fake MetaTrader 5 and fake LLM agents that sleep for a fixed latency.

Used to (a) check the whole loop runs end to end, (b) measure tick time and
count LLM calls before/after performance changes.

    py tests/tick_harness.py --symbols 8 --latency 0.5

No terminal, no network, no API key. Runs in a temporary directory.
"""

import argparse
import os
import sys
import tempfile
import threading
import time
import types
from collections import Counter
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402


def _fake_mt5_module():
    m = types.ModuleType("MetaTrader5")
    m.__getattr__ = lambda name: 0
    m.ORDER_TYPE_BUY, m.ORDER_TYPE_SELL = 0, 1
    m.symbol_info = lambda s: types.SimpleNamespace(
        point=0.00001, digits=5, trade_tick_value=1.0, trade_tick_size=0.00001,
        volume_min=0.01, volume_step=0.01, volume_max=100.0)
    m.positions_get = lambda *a, **k: []
    return m


def _bars(n, freq, seed):
    rng = np.random.default_rng(seed)
    c = 1.08 * np.exp(np.cumsum(rng.normal(0.0003, 0.0015, n)))   # gentle uptrend
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, .0008, n)))
    lo = np.minimum(o, c) * (1 - abs(rng.normal(0, .0008, n)))
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=n, freq=freq)
    df = pd.DataFrame({"time": idx, "open": o, "high": h, "low": lo, "close": c,
                       "tick_volume": rng.integers(100, 1000, n)})
    df["volume"] = df["tick_volume"]
    return df.set_index("time", drop=False)


def run_one_tick(n_symbols=6, latency=0.3, buy_every=3, desk_workers=None,
                 ticks=1, cfg_extra=None):
    """Run `ticks` loop iterations. Returns dict(tick_seconds of the first tick,
    llm_calls, orders, threads_seen, mt5_threads, error)."""
    workdir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(workdir)
    sys.modules["MetaTrader5"] = _fake_mt5_module()
    for mod in ("mt5_bridge", "position_manager", "bot"):
        sys.modules.pop(mod, None)

    import config
    import bot

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF",
               "EURGBP", "EURJPY", "GBPJPY", "XAUUSD", "AUDJPY"][:n_symbols]
    calls, orders, threads_seen = Counter(), [], set()
    mt5_threads = set()          # every thread that touched MT5 (must be MainThread)
    lock = threading.Lock()

    def _mt5(fn):
        def wrapped(*a, **k):
            with lock:
                mt5_threads.add(threading.current_thread().name)
            return fn(*a, **k)
        return wrapped

    def _agent(name, result_fn):
        def fn(*a, **k):
            with lock:
                calls[name] += 1
                threads_seen.add(threading.current_thread().name)
            time.sleep(latency)
            return result_fn(*a, **k)
        return fn

    def _pm(*a, **k):
        sym = k["ctx"].get("symbol") or ""
        i = symbols.index(sym) if sym in symbols else 0
        if buy_every and i % buy_every == 0:
            return {"action": "BUY", "confidence": 85, "reasoning": "harness",
                    "bull_bear_verdict": "BULL"}
        return {"action": "HOLD", "confidence": 40, "reasoning": "harness"}

    _raw = sys.modules["MetaTrader5"]
    _raw.symbol_info   = _mt5(_raw.symbol_info)
    _raw.positions_get = _mt5(_raw.positions_get)

    tick = {"start": None, "end": None}

    def _sleep(tick_start):
        tick["n"] = tick.get("n", 0) + 1
        if tick["end"] is None:
            tick["end"] = time.perf_counter()
        if tick["n"] >= ticks:
            raise KeyboardInterrupt   # bot.run() treats this as a clean stop

    def _place_order(**k):
        orders.append(k["symbol"])
        return {"ticket": 1000 + len(orders), "price": 1.1, "sl": 1.09, "tp": 1.12,
                "volume": k["lot_size"]}

    cfg_over = dict(SYMBOLS=symbols, LIVE_CALENDAR_ENABLED=False, CUSUM_DISABLED=True,
                    TELEGRAM_ALERTS_ENABLED=False, ALLOWED_SESSIONS=[],
                    ALLOWED_REGIMES=["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE",
                                     "BREAKOUT", "LOW_VOLATILITY", "UNKNOWN"],
                    MIN_CONFIDENCE_TO_TRADE=50, META_LABEL_MIN_PROB=0.0,
                    MAX_OPEN_POSITIONS=50, MAX_ORDERS_PER_DAY=50,
                    ENTRY_TIMING_ENABLED=False, SPREAD_FILTER_ENABLED=False,
                    SETUP_BLACKLIST_ENABLED=False, CORRELATION_GUARD_ENABLED=False,
                    DISCOVERED_SIGNALS_ENABLED=False)
    if desk_workers is not None:
        cfg_over["DESK_PARALLELISM"] = desk_workers
    cfg_over.update(cfg_extra or {})

    patches = [mock.patch.object(config, k, v, create=True) for k, v in cfg_over.items()]
    mb = bot.mt5b
    patches += [
        mock.patch.object(mb, "connect", side_effect=_mt5(lambda: True)),
        mock.patch.object(mb, "disconnect", side_effect=_mt5(lambda: None)),
        mock.patch.object(mb, "is_connected", side_effect=_mt5(lambda: True)),
        mock.patch.object(mb, "resolve_symbol", side_effect=_mt5(lambda s: s)),
        mock.patch.object(mb, "get_account_equity", side_effect=_mt5(lambda: 10_000.0)),
        mock.patch.object(mb, "get_account_balance", side_effect=_mt5(lambda: 10_000.0)),
        mock.patch.object(mb, "get_account_login", side_effect=_mt5(lambda: "12345"), create=True),
        mock.patch.object(mb, "get_all_open_positions", side_effect=_mt5(lambda *a: [])),
        mock.patch.object(mb, "get_candles", side_effect=_mt5(
            lambda s, tf, count=200, **k: _bars(count, "h" if tf != "H4" else "4h", abs(hash(s)) % 1000))),
        mock.patch.object(mb, "get_live_price", side_effect=_mt5(lambda s: None)),
        mock.patch.object(mb, "get_spread_pips", side_effect=_mt5(lambda s: 1.0)),
        mock.patch.object(mb, "get_risk_usd", side_effect=_mt5(lambda *a, **k: 10.0)),
        mock.patch.object(mb, "place_order", side_effect=_mt5(_place_order)),
        mock.patch.object(bot.llm, "init", return_value=None),
        mock.patch.object(bot, "_sleep", side_effect=_sleep),
        mock.patch.object(bot, "fetch_headlines", side_effect=_agent("headlines", lambda *a, **k: ["Dollar firm"])),
        mock.patch.object(bot.technical_analyst, "analyse", side_effect=_agent(
            "technical", lambda *a, **k: {"bias": "BULLISH", "bias_strength": 7, "summary": "h"})),
        mock.patch.object(bot.bear_analyst, "analyse", side_effect=_agent(
            "bear", lambda *a, **k: {"bear_thesis": "h", "risk_score": 3})),
        mock.patch.object(bot.news_analyst, "analyse", side_effect=_agent(
            "news", lambda *a, **k: {"macro_tone": "NEUTRAL", "news_risk": 3})),
        mock.patch.object(bot.portfolio_manager, "decide", side_effect=_agent("portfolio_manager", _pm)),
        mock.patch.object(bot.risk_mgr_agent, "assess", side_effect=_agent(
            "risk_manager", lambda *a, **k: {"trade_qualified": True, "size_multiplier": 1.0})),
    ]
    if hasattr(bot, "get_macro_context"):
        patches.append(mock.patch.object(bot, "get_macro_context", return_value={}))
    sem = types.ModuleType("memory.semantic")
    sem.embed_market_context = lambda *a, **k: None
    sem.embed_text = lambda *a, **k: None
    patches.append(mock.patch.dict(sys.modules, {"memory.semantic": sem}))

    err = None
    for p in patches:
        p.start()
    try:
        tick["start"] = time.perf_counter()
        bot.run(demo_mode=False)
    except SystemExit as e:
        err = f"SystemExit {e}"
    except Exception as e:  # pragma: no cover - reported to caller
        import traceback
        err = traceback.format_exc()
    finally:
        for p in reversed(patches):
            p.stop()
        os.chdir(old_cwd)
    end = tick["end"] or time.perf_counter()
    return {"tick_seconds": round(end - tick["start"], 2), "llm_calls": dict(calls),
            "orders": orders, "threads_seen": len(threads_seen),
            "mt5_threads": sorted(mt5_threads), "error": err}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=6)
    ap.add_argument("--latency", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    import logging
    logging.disable(logging.CRITICAL)
    print(run_one_tick(a.symbols, a.latency, desk_workers=a.workers))
