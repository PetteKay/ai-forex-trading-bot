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
