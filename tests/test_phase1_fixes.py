"""
Phase 1 regression tests — each test reproduces a verified bug.

Every test here FAILED on commit 8304ced (before the fix) and passes after it.
Runs offline: no MT5 terminal, no LLM key.

Run: py -m pytest tests/test_phase1_fixes.py -v
"""

import json
import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import numpy as np    # noqa: E402  (imported up-front: numpy cannot be re-imported)
import pandas as pd   # noqa: E402,F401

from risk_governor import RiskGovernor  # noqa: E402


def _install_fake_mt5(test, fake):
    """Swap in a fake MetaTrader5 module for one test, restoring only that key."""
    had = "MetaTrader5" in sys.modules
    old = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = fake
    for mod in ("mt5_bridge", "position_manager"):
        sys.modules.pop(mod, None)

    def restore():
        if had:
            sys.modules["MetaTrader5"] = old
        else:
            sys.modules.pop("MetaTrader5", None)
        for mod in ("mt5_bridge", "position_manager"):
            sys.modules.pop(mod, None)
    test.addCleanup(restore)


def _gov(equity, path):
    return RiskGovernor(
        starting_equity=equity, daily_loss_cap_pct=2.0, max_open_positions=3,
        max_orders_per_day=10, max_drawdown_pct=10.0, state_path=path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1: a restart (manual or crash auto-restart) erased the safety stops
# ─────────────────────────────────────────────────────────────────────────────
class TestGovernorSurvivesRestart(unittest.TestCase):
    def setUp(self):
        self.dir  = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "governor_state.json")

    def test_kill_switch_survives_same_day_restart(self):
        g = _gov(1000.0, self.path)
        g.record_trade_closed(-25.0)                       # -2.5% > 2% cap
        self.assertFalse(g.approve(975.0, 0)[0])
        g2 = _gov(975.0, self.path)                        # restart
        ok, reason = g2.approve(975.0, 0)
        self.assertFalse(ok, "Restart cleared the daily kill switch")
        self.assertIn("Kill switch", reason)

    def test_daily_loss_carries_over_restart(self):
        g = _gov(1000.0, self.path)
        g.record_trade_closed(-15.0)                       # -1.5%, under cap
        g2 = _gov(985.0, self.path)
        self.assertAlmostEqual(g2.status["daily_pnl"], -15.0)
        g2.record_trade_closed(-6.0)                       # total -2.1% of 1000
        self.assertFalse(g2.approve(979.0, 0)[0],
                         "Losses before the restart were forgotten")

    def test_daily_cap_keeps_day_anchor(self):
        g = _gov(1000.0, self.path)
        g.record_trade_closed(-15.0)
        g2 = _gov(985.0, self.path)                        # cap must stay 2% of 1000
        g2.record_trade_closed(-5.1)                       # -20.1 total > 20.0
        self.assertFalse(g2.approve(979.9, 0)[0])

    def test_drawdown_halt_survives_restart(self):
        g = _gov(1000.0, self.path)
        g.update_equity(1000.0)
        self.assertFalse(g.approve(880.0, 0)[0])           # 12% drawdown
        g2 = _gov(880.0, self.path)
        ok, reason = g2.approve(880.0, 0)
        self.assertFalse(ok, "Restart cleared the drawdown halt")
        self.assertIn("Drawdown", reason)

    def test_manual_reset_clears_drawdown_halt(self):
        g = _gov(1000.0, self.path)
        g.approve(880.0, 0)
        g2 = _gov(880.0, self.path)
        g2.reset_drawdown_halt(880.0)
        self.assertTrue(g2.approve(880.0, 0)[0])
        g3 = _gov(880.0, self.path)                        # reset is saved too
        self.assertTrue(g3.approve(880.0, 0)[0])

    def test_new_day_clears_daily_but_not_drawdown(self):
        g = _gov(1000.0, self.path)
        g.record_trade_closed(-25.0)
        with open(self.path) as f:
            st = json.load(f)
        st["today"] = "2000-01-01"                         # pretend saved yesterday
        with open(self.path, "w") as f:
            json.dump(st, f)
        g2 = _gov(975.0, self.path)
        self.assertTrue(g2.approve(975.0, 0)[0])
        self.assertEqual(g2.status["daily_pnl"], 0.0)

    def test_corrupt_state_file_fails_safe(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        g = _gov(1000.0, self.path)
        self.assertFalse(g.approve(1000.0, 0)[0],
                         "Unreadable state must not silently re-enable trading")

    def test_no_state_path_keeps_old_in_memory_behaviour(self):
        g = _gov(1000.0, None)
        self.assertTrue(g.approve(1000.0, 0)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2: trades closed by the kill switch never had their P&L recorded
# ─────────────────────────────────────────────────────────────────────────────
class TestKillSwitchKeepsRecordsForPnL(unittest.TestCase):
    def _fake_mt5(self):
        import types
        m = types.ModuleType("MetaTrader5")
        m.__getattr__ = lambda name: 0          # any other MT5 constant
        m.ORDER_TYPE_BUY = 0
        pos = types.SimpleNamespace(type=0, symbol="EURUSD", volume=0.01)
        m.positions_get = lambda ticket=None: [pos]
        return m

    def test_close_all_leaves_tickets_for_pnl_booking(self):
        import types
        from unittest import mock
        fake = self._fake_mt5()
        _install_fake_mt5(self, fake)
        import position_manager
        with mock.patch.object(position_manager.mt5b, "close_position", return_value=True) as cp:
            open_trades = {111: object(), 222: object()}
            position_manager.PositionManager(types.SimpleNamespace()).close_all(
                open_trades, types.SimpleNamespace(MAGIC_NUMBER=1))
        self.assertEqual(cp.call_count, 2, "Both positions must be closed")
        self.assertEqual(set(open_trades), {111, 222},
                         "close_all removed the records, so their P&L could never be booked")

    def test_closed_trade_detection_runs_before_kill_guard(self):
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "bot.py")).read()
        detect = src.index("# ── 3. Detect closed trades")
        guard  = src.index("# ── Kill-switch guard")
        self.assertLess(detect, guard,
                        "Kill guard skips the tick before closed trades are booked")


# ─────────────────────────────────────────────────────────────────────────────
# Bug 3: signals were computed on the still-forming candle
# MQL5 docs: bar index 0 is the current (unfinished) bar.
# ─────────────────────────────────────────────────────────────────────────────
class TestClosedCandlesOnly(unittest.TestCase):
    def _bridge_with(self, calls):
        import types
        m = types.ModuleType("MetaTrader5")
        m.__getattr__ = lambda name: 0
        m.TIMEFRAME_H1 = 16385

        def copy_rates_from_pos(symbol, tf, start, count):
            calls.append(start)
            dt = np.dtype([("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                           ("close", "f8"), ("tick_volume", "i8")])
            return np.array([(1_700_000_000 + i * 3600, 1, 1, 1, 1, 10) for i in range(count)], dtype=dt)
        m.copy_rates_from_pos = copy_rates_from_pos
        _install_fake_mt5(self, m)
        import mt5_bridge
        return mt5_bridge

    def test_default_skips_forming_bar(self):
        calls = []
        b = self._bridge_with(calls)
        df = b.get_candles("EURUSD", "H1", count=5)
        self.assertEqual(calls, [1], "get_candles still asks MT5 for bar 0 (the forming candle)")
        self.assertEqual(len(df), 5)

    def test_forming_bar_only_on_request(self):
        calls = []
        b = self._bridge_with(calls)
        b.get_candles("EURUSD", "H1", count=5, include_forming=True)
        self.assertEqual(calls, [0])


# ─────────────────────────────────────────────────────────────────────────────
# Bugs 4-7: meta-label gate
# ─────────────────────────────────────────────────────────────────────────────
def _fake_trades(n, seed=0, start=None, direction_fn=None, hours_apart=6):
    from datetime import datetime, timedelta, timezone
    rng = np.random.default_rng(seed)
    start = start or datetime.now(timezone.utc) - timedelta(hours=hours_apart * n + 1)
    out = []
    for i in range(n):
        t0 = start + timedelta(hours=hours_apart * i)
        d  = direction_fn(i) if direction_fn else ("BUY" if i % 2 else "SELL")
        r  = float(rng.normal())
        out.append({
            "feature_vector": list(rng.random(21)), "symbol": "EURUSD", "direction": d,
            "entry_time": t0.isoformat(), "exit_time": (t0 + timedelta(hours=5)).isoformat(),
            "timestamp": t0.isoformat(), "outcome": "WIN" if r > 0 else "LOSS",
            "r_multiple": r, "pnl_pct": r, "regime": "TRENDING_UP",
        })
    return out


class TestMetaLabelTrainsOnSmallSamples(unittest.TestCase):
    """Bug 4: TimeSeriesSplit(gap=24 rows) failed for every N from 30 to ~147."""

    def test_trains_across_small_sample_sizes(self):
        from ml.meta_label import MetaLabelGate
        failed = []
        for n in (30, 45, 60, 80, 100, 120, 144):
            g = MetaLabelGate()
            g.train(_fake_trades(n, seed=n), training_lookback_days=0)
            if not g._trained:
                failed.append(n)
        self.assertEqual(failed, [], f"ML gate failed to train at N={failed}")

    def test_purged_splits_never_overlap_labels(self):
        from datetime import datetime, timedelta, timezone
        from ml.meta_label import _purged_time_splits
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        starts = [t + timedelta(hours=3 * i) for i in range(80)]
        ends   = [s + timedelta(hours=30) for s in starts]          # long-held trades
        y = np.array([i % 2 for i in range(80)])
        splits = _purged_time_splits(starts, ends, y, 4)
        self.assertGreaterEqual(len(splits), 2)
        for tr, te in splits:
            first_test = starts[te[0]]
            self.assertTrue(all(ends[i] < first_test for i in tr),
                            "A training label is still open when the test block starts")


class TestMetaLabelKnowsTheSide(unittest.TestCase):
    """Bug 5: the gate could not tell a BUY from a SELL in the same market state."""

    def test_same_market_different_side_gives_different_probability(self):
        from ml.meta_label import MetaLabelGate
        rng = np.random.default_rng(7)
        trades = _fake_trades(160, seed=7)
        for tr in trades:                         # BUYs win, SELLs lose, same features
            tr["feature_vector"] = list(rng.random(21))
            tr["r_multiple"] = 1.0 if tr["direction"] == "BUY" else -1.0
            tr["outcome"] = "WIN" if tr["direction"] == "BUY" else "LOSS"
        g = MetaLabelGate()
        self.assertTrue(g.train(trades, training_lookback_days=0))
        fv = list(rng.random(21))
        ctx = {"regime": {"regime": "TRENDING_UP"}}
        p_buy  = g.predict(fv, ctx, direction="BUY")["probability"]
        p_sell = g.predict(fv, ctx, direction="SELL")["probability"]
        self.assertGreater(p_buy - p_sell, 0.2,
                           f"Side ignored: BUY={p_buy:.3f} SELL={p_sell:.3f}")

    def test_training_row_contains_side(self):
        from ml.meta_label import _build_training_row
        base = _fake_trades(1)[0]
        buy  = _build_training_row({**base, "direction": "BUY"})[0]
        sell = _build_training_row({**base, "direction": "SELL"})[0]
        self.assertNotEqual(buy, sell)
        self.assertEqual((buy[-1], sell[-1]), (1.0, -1.0))


class TestMetaLabelUsesEntryTime(unittest.TestCase):
    """Bug 6: training took hour/weekday from the lesson save time (trade close)."""

    def test_training_time_features_follow_entry_time(self):
        from ml.meta_label import _build_training_row, _build_extended_features
        m = {**_fake_trades(1)[0],
             "entry_time": "2026-03-02T08:00:00+00:00",     # Monday 08:00 (entry)
             "timestamp":  "2026-03-03T20:00:00+00:00"}     # Tuesday 20:00 (saved)
        row = _build_training_row(m)[0]
        live = _build_extended_features(m["feature_vector"], {}, regime_override=row[-5],
                                        side=1.0 if m["direction"] == "BUY" else -1.0,
                                        when="2026-03-02T08:00:00+00:00")
        np.testing.assert_allclose(row[21:25], live[21:25], atol=1e-9)


class TestEveryClosedTradeIsTrainingData(unittest.TestCase):
    """Bug 7: only lessons the reflection LLM kept (and pruning spared) were trained on."""

    def test_unsaved_lessons_still_train(self):
        from memory.store import TradeMemoryStore
        d = tempfile.mkdtemp()
        store = TradeMemoryStore(os.path.join(d, "m.db"), usefulness_threshold=0.9)
        for i, tr in enumerate(_fake_trades(5)):
            store.save_ml_sample({**tr, "ticket": 1000 + i})
        rows = store.get_all_for_training()
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r["entry_time"] for r in rows))

    def test_saved_lesson_not_counted_twice(self):
        from memory.store import TradeMemoryStore
        d = tempfile.mkdtemp()
        store = TradeMemoryStore(os.path.join(d, "m.db"), usefulness_threshold=0.1)
        tr = _fake_trades(1)[0]
        store.save_ml_sample({**tr, "ticket": 1})
        store.save_ml_sample({**tr, "ticket": 1})                  # idempotent
        lesson = {**tr, "entry_price": 1.0, "exit_price": 1.1, "fingerprint": "x",
                  "lesson": "test", "usefulness_score": 0.9, "ml_sampled": 1}
        from unittest import mock
        with mock.patch.dict(sys.modules, {"memory.semantic": None}):
            self.assertTrue(store.save(lesson))
        self.assertEqual(len(store.get_all_for_training()), 1)

    def test_old_memories_still_used(self):
        from memory.store import TradeMemoryStore
        d = tempfile.mkdtemp()
        store = TradeMemoryStore(os.path.join(d, "m.db"), usefulness_threshold=0.1)
        tr = _fake_trades(1)[0]
        lesson = {**tr, "entry_price": 1.0, "exit_price": 1.1, "fingerprint": "x",
                  "lesson": "old", "usefulness_score": 0.9}         # pre-fix: ml_sampled 0
        from unittest import mock
        with mock.patch.dict(sys.modules, {"memory.semantic": None}):
            store.save(lesson)
        self.assertEqual(len(store.get_all_for_training()), 1)
