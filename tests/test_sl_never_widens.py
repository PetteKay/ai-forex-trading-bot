"""
INV-PL-1 (specs/PositionLifecycle.tla, INV_PL1_SLNeverWidens): a BUY's stop loss
never moves down and a SELL's never moves up.

TLC counterexample that this test reproduces in the real position manager:
trailing stop moves the SL past entry, then TP-ladder tier 1 fires and moved the
SL "to break-even" — back below the trailed SL, widening the risk.

Run: py -m pytest tests/test_sl_never_widens.py -v
"""
import importlib.util
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _fake_mt5():
    spec = importlib.util.spec_from_file_location(
        "MetaTrader5", os.path.join(ROOT, "tests", "fake_mt5", "MetaTrader5.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class SLNeverWidens(unittest.TestCase):
    def setUp(self):
        self.had = sys.modules.get("MetaTrader5")
        self.mt5 = sys.modules["MetaTrader5"] = _fake_mt5()
        sys.modules.pop("position_manager", None)
        import position_manager
        self.pm_mod = position_manager

    def tearDown(self):
        if self.had is not None:
            sys.modules["MetaTrader5"] = self.had
        else:
            sys.modules.pop("MetaTrader5", None)
        sys.modules.pop("position_manager", None)

    def _run(self, is_buy, entry, sl0, price, atr):
        """One manage() pass on a position already +35 pips with a 30-pip initial risk."""
        pos = types.SimpleNamespace(
            ticket=7, symbol="EURUSD", price_open=entry, price_current=price, sl=sl0,
            tp=0.0, volume=0.30, type=0 if is_buy else 1, profit=100.0)
        history = [sl0]

        def positions_get(ticket=None):
            return (pos,)

        def modify(ticket, symbol, new_sl):
            pos.sl = new_sl
            history.append(new_sl)
            return True

        def close(ticket, symbol, vol, pos_type, magic):
            pos.volume = round(pos.volume - vol, 2)
            return True
        cfg = types.SimpleNamespace(
            BREAKEVEN_PIPS=1000, TRAIL_START_PIPS=20, ATR_TRAIL_MULT=1.0,
            TP_LADDER_ENABLED=True, TP_LADDER_TIER1_R=1.0, TP_LADDER_TIER2_R=2.0,
            TP_LADDER_TIER1_PCT=33, TP_LADDER_TIER2_PCT=33, MAGIC_NUMBER=1,
            MAX_HOLD_HOURS=48, MAX_POSITION_LOSS_USD=None, LOT_SIZE=0.01)
        record = types.SimpleNamespace(atr=atr, volume=0.30, sl=sl0, entry_price=entry,
                                       entry_time=datetime.now(timezone.utc))
        with mock.patch.object(self.mt5, "positions_get", positions_get), \
             mock.patch.object(self.pm_mod.mt5b, "modify_position_sl", modify), \
             mock.patch.object(self.pm_mod.mt5b, "close_position", close):
            self.pm_mod.PositionManager(cfg).manage({7: record}, 10_000)
        return history, pos

    def test_buy_tier1_does_not_pull_trailed_sl_back_to_breakeven(self):
        hist, pos = self._run(True, entry=1.10000, sl0=1.09700, price=1.10350, atr=0.00100)
        self.assertLess(pos.volume, 0.30, "tier 1 did not fire — test setup wrong")
        self.assertGreater(len(hist), 1, "trailing did not move the SL — test setup wrong")
        for a, b in zip(hist, hist[1:]):
            self.assertGreaterEqual(b, a, f"BUY SL widened: {hist}")
        self.assertAlmostEqual(pos.sl, 1.10250)

    def test_sell_tier1_does_not_pull_trailed_sl_back_to_breakeven(self):
        hist, pos = self._run(False, entry=1.10000, sl0=1.10300, price=1.09650, atr=0.00100)
        self.assertLess(pos.volume, 0.30)
        self.assertGreater(len(hist), 1)
        for a, b in zip(hist, hist[1:]):
            self.assertLessEqual(b, a, f"SELL SL widened: {hist}")
        self.assertAlmostEqual(pos.sl, 1.09750)

    def test_breakeven_still_applied_when_it_tightens(self):
        # no trailing yet (TRAIL_START far away): tier 1 must still move SL to break-even
        hist, pos = self._run(True, entry=1.10000, sl0=1.09700, price=1.10310, atr=0.00500)
        self.assertLess(pos.volume, 0.30)
        self.assertAlmostEqual(pos.sl, 1.10010)


if __name__ == "__main__":
    unittest.main()
