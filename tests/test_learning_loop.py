"""
Learning loop tests: what a closed trade teaches the bot.

bot.py — runs the REAL loop (tests/tick_harness.py): tick 1 opens a trade,
         tick 2 sees it closed → ML sample, reflection lesson, blacklist.
MCP    — see TestMCPLearning (fake MT5 terminal).
"""
import json
import logging
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.CRITICAL)

from tick_harness import run_one_tick  # noqa: E402

LESSON = {"lesson": "EURUSD H1 trend entries after London open held well",
          "usefulness_score": 0.9, "category": "ENTRY_TIMING", "pattern_name": "London trend"}


def _rows(workdir, db, sql):
    con = sqlite3.connect(os.path.join(workdir, db))
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql).fetchall()]
    finally:
        con.close()


def _close_all(n, cfg, positions):
    if n == 1:
        positions.clear()


class TestBotCloseLearning(unittest.TestCase):
    def _run(self, pnl):
        r = run_one_tick(1, 0.0, buy_every=1, desk_workers=1, ticks=2, reflect_result=LESSON,
                         close_pnl=pnl, on_tick_end=_close_all,
                         cfg_extra={"SETUP_BLACKLIST_ENABLED": True})
        self.assertIsNone(r["error"], r["error"])
        self.assertEqual(r["orders"], ["EURUSD"])
        return r

    def test_closed_trade_feeds_ml_lessons_and_journal(self):
        r = self._run(-5.0)
        w = r["workdir"]
        import config
        ml = _rows(w, config.MEMORY_DB_PATH, "SELECT * FROM ml_samples")
        self.assertEqual(len(ml), 1)
        self.assertEqual((ml[0]["symbol"], ml[0]["direction"]), ("EURUSD", "BUY"))
        self.assertEqual(len(json.loads(ml[0]["feature_vector"])), 21)
        self.assertAlmostEqual(ml[0]["r_multiple"], -0.5)          # -5 / risk 10
        lessons = _rows(w, config.MEMORY_DB_PATH, "SELECT * FROM memories")
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["lesson"], LESSON["lesson"])
        self.assertEqual(lessons[0]["ml_sampled"], 1)
        self.assertTrue(lessons[0]["entry_time"])
        self.assertEqual(r["llm_calls"]["reflection"], 1)
        trades = _rows(w, config.JOURNAL_DB_PATH, "SELECT * FROM trades")
        self.assertEqual(trades[0]["status"], "CLOSED")
        bl = _rows(w, config.JOURNAL_DB_PATH, "SELECT * FROM setup_blacklist")
        self.assertEqual(sum(b.get("consecutive_losses", b.get("loss_count", 0)) or 0 for b in bl), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestEntrySnapshot(unittest.TestCase):
    def test_snapshot_round_trip_keeps_features(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import numpy as np
        from memory.trade_learning import json_safe_context
        from strategy.indicators import feature_vector
        ctx = {"price": np.float64(1.08), "rsi": np.float64(55.2), "fast_above_slow": np.bool_(True),
               "price_above_trend": True, "macd_positive": np.bool_(False), "atr_percentile": 64,
               "bb_pct": 0.4, "_df": object(), "regime": {"regime": "TRENDING_UP"}}
        self.assertEqual(feature_vector(json_safe_context(ctx)), feature_vector(ctx))

    def test_bot_stores_snapshot_and_restores_it_after_restart(self):
        import config
        r1 = run_one_tick(1, 0.0, buy_every=1, desk_workers=1, ticks=1)
        self.assertEqual(r1["orders"], ["EURUSD"])
        snap = _rows(r1["workdir"], config.MEMORY_DB_PATH, "SELECT * FROM open_trade_context")
        self.assertEqual(len(snap), 1)
        ticket = snap[0]["ticket"]
        self.assertTrue(json.loads(snap[0]["context"]).get("rsi") is not None)
        # Restart: the position is still open in MT5; tick 1 adopts it, then it closes.
        pos = [{"ticket": ticket, "symbol": "EURUSD", "type": "BUY", "volume": 0.01,
                "price": 1.1, "sl": 1.09, "tp": 1.12, "profit": 0.0, "time": 0}]
        r2 = run_one_tick(1, 0.0, buy_every=0, desk_workers=1, ticks=2, workdir=r1["workdir"],
                          start_positions=pos, on_tick_end=_close_all, close_pnl=-5.0,
                          reflect_result=LESSON)
        self.assertIsNone(r2["error"], r2["error"])
        ml = _rows(r1["workdir"], config.MEMORY_DB_PATH, "SELECT * FROM ml_samples")
        self.assertEqual([m["ticket"] for m in ml], [ticket],
                         "adopted trade lost its entry context → no ML sample")
        self.assertAlmostEqual(ml[0]["r_multiple"], -0.5)   # risk_usd restored (10)
        self.assertEqual(_rows(r1["workdir"], config.MEMORY_DB_PATH,
                               "SELECT * FROM open_trade_context"), [])
