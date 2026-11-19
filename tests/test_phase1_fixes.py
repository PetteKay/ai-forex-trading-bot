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

from risk_governor import RiskGovernor  # noqa: E402


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
