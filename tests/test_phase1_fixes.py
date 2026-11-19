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
