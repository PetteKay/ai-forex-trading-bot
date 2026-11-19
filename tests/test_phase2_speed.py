"""
Phase 2 speed tests. Run the REAL bot.run() loop for one tick (tests/tick_harness.py)
with a fake MT5 and fake LLM agents that each take a fixed time.

Run: py -m pytest tests/test_phase2_speed.py -v
"""
import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.CRITICAL)

from tick_harness import run_one_tick  # noqa: E402

N, LAT = 6, 0.15
EXPECTED_ORDERS = ["EURUSD", "AUDUSD"]     # PM says BUY on every 3rd symbol


class TestRiskManagerOnlyForCandidates(unittest.TestCase):
    def test_same_trades_fewer_risk_manager_calls(self):
        r = run_one_tick(N, LAT)
        self.assertIsNone(r["error"], r["error"])
        self.assertEqual(r["orders"], EXPECTED_ORDERS)
        self.assertEqual(r["llm_calls"]["risk_manager"], len(EXPECTED_ORDERS),
                         "Risk Manager LLM ran for HOLDs / blocked trades")


class TestParallelDesks(unittest.TestCase):
    def test_parallel_desks_same_decisions_and_faster(self):
        seq = run_one_tick(N, LAT, desk_workers=1)
        par = run_one_tick(N, LAT, desk_workers=3)
        for r in (seq, par):
            self.assertIsNone(r["error"], r["error"])
            self.assertEqual(r["orders"], EXPECTED_ORDERS)
        self.assertEqual(seq["llm_calls"], par["llm_calls"])
        self.assertLess(par["tick_seconds"], seq["tick_seconds"] * 0.75,
                        f"parallel {par['tick_seconds']}s vs sequential {seq['tick_seconds']}s")

    def test_mt5_only_touched_from_main_thread(self):
        r = run_one_tick(N, LAT, desk_workers=4)
        self.assertIsNone(r["error"], r["error"])
        self.assertEqual(r["mt5_threads"], ["MainThread"])


class TestNewsRunsAlongsideTechnical(unittest.TestCase):
    def test_desk_critical_path(self):
        """One desk: Technical+Bear (2 calls) overlaps Headlines+News (2 calls) → ~3 calls, not 5."""
        import time, types
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agents.desk import run_desk
        lat = 0.2
        slow = lambda result: (lambda *a, **k: (time.sleep(lat), result)[1])
        agents = {
            "technical": types.SimpleNamespace(analyse=slow({"bias": "BULLISH"})),
            "bear": types.SimpleNamespace(analyse=slow({"risk_score": 3})),
            "news": types.SimpleNamespace(analyse=slow({"macro_tone": "NEUTRAL"})),
            "portfolio_manager": types.SimpleNamespace(decide=slow({"action": "HOLD"})),
        }
        job = {"symbol": "EURUSD", "ctx": {}, "regime_info": {}, "memories": [],
               "extra_signals": {}, "upcoming_events": []}
        t = time.perf_counter()
        out = run_desk(job, types.SimpleNamespace(), agents, slow(["h"]))
        took = time.perf_counter() - t
        self.assertEqual(out["pm"]["action"], "HOLD")
        self.assertEqual(out["live_headlines"], ["h"])
        # sequential would be 5 × 0.2 = 1.0 s; overlapped critical path is 3 × 0.2 = 0.6 s
        self.assertLess(took, 0.8, f"desk took {took:.2f}s — News not overlapping Technical")


class TestSharedFetchesUnderParallelDesks(unittest.TestCase):
    def _reset_news(self):
        from data import news_feed
        news_feed._cache_ts, news_feed._cache_headlines, news_feed._fail_ts = None, [], None
        return news_feed

    def test_headlines_fetched_once_for_many_threads(self):
        import threading, time
        from unittest import mock
        nf = self._reset_news()
        n_fetch = []
        def slow_parse(url):
            n_fetch.append(url); time.sleep(0.2); return ["Dollar firm"]
        with mock.patch.object(nf, "_parse_rss", side_effect=slow_parse):
            ts = [threading.Thread(target=nf.fetch_headlines, args=(s,)) for s in
                  ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD", "USDCAD")]
            [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(len(n_fetch), len(nf._RSS_FEEDS), "feeds were fetched more than once")

    def test_failed_feeds_back_off(self):
        from unittest import mock
        from urllib.error import URLError
        nf = self._reset_news()
        with mock.patch.object(nf, "_parse_rss", side_effect=URLError("down")) as p:
            for s in ("EURUSD", "GBPUSD", "USDJPY"):
                self.assertEqual(nf.fetch_headlines(s), [])
        self.assertEqual(p.call_count, len(nf._RSS_FEEDS),
                         "every symbol retried every dead feed")

    def test_macro_fetched_once_and_failure_backs_off(self):
        import threading, time
        from unittest import mock
        from data import macro_context as mc
        import data.process_isolation as pi
        mc._cache, mc._cache_ts = {}, 0.0
        n = []
        def iso(fn, *a, **k):
            n.append(1); time.sleep(0.2); return {}
        with mock.patch.object(pi, "run_isolated", side_effect=iso):
            ts = [threading.Thread(target=mc.get_macro_context, args=("EURUSD",)) for _ in range(5)]
            [t.start() for t in ts]; [t.join() for t in ts]
            mc.get_macro_context("GBPUSD")
        self.assertEqual(len(n), 1, f"macro fetch ran {len(n)} times")


class TestOneEvaluationPerCandle(unittest.TestCase):
    def test_bar_gate(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from strategy.bar_gate import BarGate
        g = BarGate()
        self.assertFalse(g.already_evaluated("EURUSD", 1))
        g.mark("EURUSD", 1)
        self.assertTrue(g.already_evaluated("EURUSD", 1))
        self.assertFalse(g.already_evaluated("EURUSD", 2))     # new candle
        g.retry("EURUSD")
        self.assertFalse(g.already_evaluated("EURUSD", 1))

    def test_same_candle_not_reevaluated_across_ticks(self):
        # 12 ticks (12 minutes) on one unchanged closed candle, CUSUM on with the
        # previous defaults (force evaluation every 10 ticks).
        r = run_one_tick(3, 0.0, buy_every=0, ticks=12, cfg_extra={
            "CUSUM_DISABLED": False, "CUSUM_MAX_BARS": 10, "CUSUM_MIN_BARS": 1})
        self.assertIsNone(r["error"], r["error"])
        self.assertEqual(r["llm_calls"]["technical"], 3,
                         "agent desk re-ran on a candle it had already evaluated")


class TestUnusedDesksReleaseTheirCandle(unittest.TestCase):
    """Codex review on #2: a break at max positions must not lock later symbols out
    of the rest of the candle."""

    def test_symbols_cut_off_by_max_positions_are_evaluated_when_capacity_frees(self):
        def close_the_position(n, cfg, positions):
            if n == 1:
                positions.clear()                # EURUSD hits TP → capacity frees up
        r = run_one_tick(4, 0.05, buy_every=1, desk_workers=4, ticks=2,
                         cfg_extra={"MAX_OPEN_POSITIONS": 1},
                         on_tick_end=close_the_position)
        self.assertIsNone(r["error"], r["error"])
        # tick 1: EURUSD placed, limit hit. tick 2: EURUSD closed → the next
        # symbol that was cut off (GBPUSD) is evaluated and placed.
        self.assertEqual(r["orders"], ["EURUSD", "GBPUSD"],
                         "symbols skipped after the break stayed locked for the candle")

    def test_no_desks_when_already_at_max_positions(self):
        r = run_one_tick(4, 0.05, buy_every=1, desk_workers=4, ticks=2,
                         cfg_extra={"MAX_OPEN_POSITIONS": 1})
        self.assertIsNone(r["error"], r["error"])
        self.assertEqual(r["orders"], ["EURUSD"])
        # tick 1: desks for up to 4 symbols; tick 2: at the limit, none
        self.assertLessEqual(r["llm_calls"].get("technical", 0), 4)
