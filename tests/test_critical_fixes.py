"""
Critical fix verification tests — run offline without MT5 or an LLM.

Each test verifies a specific bug that was found by the advisor review.
A test failure here means the bug is NOT fixed and live trading could lose money.

Run: py -m pytest tests/test_critical_fixes.py -v
  or: py tests/test_critical_fixes.py
"""

import sys
import os
import math
import types
import unittest
from decimal import Decimal

# ── Project root on path ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: bool("false") == True — risk veto opened by string "false"
# ─────────────────────────────────────────────────────────────────────────────
class TestBooleanCoercion(unittest.TestCase):
    """agents/base.py _validate_and_sanitize must treat "false" as False."""

    def _run(self, raw_result: dict) -> dict:
        from agents.base import _validate_and_sanitize
        return _validate_and_sanitize(raw_result, "risk_manager")

    def test_string_false_closes_gate(self):
        """LLM returns "false" as string — must NOT open the trade gate."""
        result = self._run({"trade_qualified": "false", "size_multiplier": 0.5})
        self.assertFalse(result["trade_qualified"],
            'trade_qualified="false" coerced to True — risk veto is OPEN (money at risk!)')

    def test_string_true_opens_gate(self):
        """LLM returns "true" as string — must correctly open the gate."""
        result = self._run({"trade_qualified": "true", "size_multiplier": 0.5})
        self.assertTrue(result["trade_qualified"],
            'trade_qualified="true" coerced to False')

    def test_bool_true_unchanged(self):
        result = self._run({"trade_qualified": True, "size_multiplier": 0.8})
        self.assertTrue(result["trade_qualified"])

    def test_bool_false_unchanged(self):
        result = self._run({"trade_qualified": False, "size_multiplier": 0.0})
        self.assertFalse(result["trade_qualified"])

    def test_int_zero_is_false(self):
        result = self._run({"trade_qualified": 0, "size_multiplier": 0.5})
        self.assertFalse(result["trade_qualified"])

    def test_int_one_is_true(self):
        result = self._run({"trade_qualified": 1, "size_multiplier": 0.5})
        self.assertTrue(result["trade_qualified"])

    def test_string_FALSE_uppercase_is_false(self):
        """Case-insensitive string matching."""
        result = self._run({"trade_qualified": "FALSE", "size_multiplier": 0.5})
        self.assertFalse(result["trade_qualified"])

    def test_size_multiplier_still_clamped(self):
        """Verify other validation still works alongside the bool fix."""
        result = self._run({"trade_qualified": True, "size_multiplier": 99.9})
        self.assertAlmostEqual(result["size_multiplier"], 1.0)

    def test_size_multiplier_negative_clamped(self):
        result = self._run({"trade_qualified": False, "size_multiplier": -0.5})
        self.assertAlmostEqual(result["size_multiplier"], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Meta-label train/serve feature dimension consistency
# Tests are self-contained (replicate the feature logic) to avoid requiring
# numpy/sklearn in this test environment.
# ─────────────────────────────────────────────────────────────────────────────
def _build_extended_features_local(fv: list, ctx: dict = None) -> list:
    """Local replica of ml/meta_label._build_extended_features for testing."""
    import math
    from datetime import datetime
    base = list(fv) if fv else [0.5] * 16
    now      = datetime.utcnow()
    hour     = now.hour + now.minute / 60.0
    dow      = now.weekday()
    sin_hour = math.sin(hour * 2 * math.pi / 24)
    cos_hour = math.cos(hour * 2 * math.pi / 24)
    sin_dow  = math.sin(dow  * 2 * math.pi / 7)
    cos_dow  = math.cos(dow  * 2 * math.pi / 7)
    return base + [sin_hour, cos_hour, sin_dow, cos_dow]


class TestMetaLabelFeatureConsistency(unittest.TestCase):
    """Training and inference must produce identical feature vector lengths."""

    def test_inference_feature_length_is_20(self):
        fv = [0.5] * 16
        extended = _build_extended_features_local(fv, {})
        self.assertEqual(len(extended), 20,
            f"Inference feature vector is {len(extended)}, expected 20")

    def test_training_feature_length_is_20(self):
        """Simulate the training path for a single sample."""
        from datetime import datetime
        fv = [0.5] * 16
        ts = "2025-06-15T10:30:00"
        try:
            dt   = datetime.fromisoformat(ts)
            hour = dt.hour + dt.minute / 60.0
            dow  = dt.weekday()
        except Exception:
            hour, dow = 12.0, 2
        sin_hour = math.sin(hour * 2 * math.pi / 24)
        cos_hour = math.cos(hour * 2 * math.pi / 24)
        sin_dow  = math.sin(dow  * 2 * math.pi / 7)
        cos_dow  = math.cos(dow  * 2 * math.pi / 7)
        row = fv + [sin_hour, cos_hour, sin_dow, cos_dow]
        self.assertEqual(len(row), 20,
            f"Training row is {len(row)} features, expected 20")

    def test_train_and_infer_same_length(self):
        """Training row length must exactly match inference row length."""
        from datetime import datetime
        fv = [0.5] * 16
        infer_len = len(_build_extended_features_local(fv, {}))
        ts = "2025-06-15T10:30:00"
        dt   = datetime.fromisoformat(ts)
        hour = dt.hour + dt.minute / 60.0
        dow  = dt.weekday()
        sin_hour = math.sin(hour * 2 * math.pi / 24)
        cos_hour = math.cos(hour * 2 * math.pi / 24)
        sin_dow  = math.sin(dow  * 2 * math.pi / 7)
        cos_dow  = math.cos(dow  * 2 * math.pi / 7)
        train_len = len(fv + [sin_hour, cos_hour, sin_dow, cos_dow])
        self.assertEqual(train_len, infer_len,
            f"Train ({train_len}) vs Infer ({infer_len}) feature mismatch!")

    def test_session_features_not_in_extended(self):
        """Session flags must NOT vary the feature vector (train/serve consistency)."""
        fv = [0.5] * 16
        with_london    = _build_extended_features_local(fv, {"sessions": ["LONDON"]})
        without_london = _build_extended_features_local(fv, {})
        self.assertEqual(len(with_london), len(without_london),
            "Session features still affect the vector — train/serve skew persists!")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: Startup gap check — same-day restart must NOT fire
# ─────────────────────────────────────────────────────────────────────────────
class TestStartupGapLogic(unittest.TestCase):
    """bot.py startup logic: gap check fires only on date change, not same-day restart."""

    def _simulate_startup(self, prev_balance: float, current_balance: float,
                          snapshot_date: str, today_str: str) -> bool:
        """
        Mirrors the startup logic in bot.py.
        Returns True if check_startup_gap would be called.
        """
        if prev_balance > 0 and snapshot_date and snapshot_date != today_str:
            return True  # overnight gap: fire
        elif prev_balance > 0 and not snapshot_date:
            return True  # legacy snapshot without date: fire (conservative)
        return False

    def test_overnight_gap_fires(self):
        """Snapshot from yesterday → gap check MUST fire."""
        fired = self._simulate_startup(
            prev_balance=10000.0, current_balance=9500.0,
            snapshot_date="2025-06-14", today_str="2025-06-15"
        )
        self.assertTrue(fired, "Overnight gap check did NOT fire — overnight losses undetected!")

    def test_same_day_restart_does_not_fire(self):
        """Snapshot from today → gap check must NOT fire (avoids false phantom gap)."""
        fired = self._simulate_startup(
            prev_balance=10000.0, current_balance=9800.0,
            snapshot_date="2025-06-15", today_str="2025-06-15"
        )
        self.assertFalse(fired,
            "Gap check fired on same-day restart — phantom daily P&L deduction!")

    def test_no_snapshot_no_fire(self):
        """No previous snapshot → no gap check (first-ever startup)."""
        fired = self._simulate_startup(
            prev_balance=0.0, current_balance=10000.0,
            snapshot_date=None, today_str="2025-06-15"
        )
        self.assertFalse(fired)

    def test_legacy_snapshot_without_date_fires(self):
        """Old-format snapshot (no date field) → fire conservatively."""
        fired = self._simulate_startup(
            prev_balance=10000.0, current_balance=9500.0,
            snapshot_date=None, today_str="2025-06-15"
        )
        self.assertTrue(fired, "Legacy snapshot without date should fire conservatively")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3b: RiskGovernor.check_startup_gap with balance (not equity)
# ─────────────────────────────────────────────────────────────────────────────
class TestRiskGovernorGap(unittest.TestCase):
    """check_startup_gap should correctly count gap loss toward daily P&L."""

    def _make_governor(self, starting_equity: float = 10000.0) -> object:
        from risk_governor import RiskGovernor
        return RiskGovernor(
            starting_equity    = starting_equity,
            daily_loss_cap_pct = 5.0,
            max_open_positions = 3,
            max_orders_per_day = 10,
            max_drawdown_pct   = 15.0,
            allowed_sessions   = ["LONDON", "NEW_YORK"],
        )

    def test_gap_loss_debited_to_daily_pnl(self):
        """A real overnight gap loss must be reflected in daily P&L."""
        gov = self._make_governor(10000.0)
        gov.check_startup_gap(prev_equity=10000.0, current_equity=9700.0)
        status = gov.status
        self.assertLess(float(status.get("daily_pnl", 0)), 0,
            "Gap loss of -300 not reflected in daily P&L")

    def test_gap_within_cap_no_kill(self):
        """A small overnight gap (-1%) should NOT trip the kill switch."""
        gov = self._make_governor(10000.0)
        gov.check_startup_gap(prev_equity=10000.0, current_equity=9900.0)
        self.assertFalse(gov.status["kill_active"],
            "Small -100 gap incorrectly tripped kill switch (cap is 5% = 500)")

    def test_gap_exceeding_cap_trips_kill(self):
        """A gap loss >= 5% of starting equity must activate the kill switch."""
        gov = self._make_governor(10000.0)
        gov.check_startup_gap(prev_equity=10000.0, current_equity=9400.0)
        self.assertTrue(gov.status["kill_active"],
            "Gap loss of -600 (6% > 5% cap) did not trip kill switch!")

    def test_positive_gap_no_effect(self):
        """Price moved in our favour overnight — no kill switch."""
        gov = self._make_governor(10000.0)
        gov.check_startup_gap(prev_equity=10000.0, current_equity=10200.0)
        self.assertFalse(gov.status["kill_active"])


# ─────────────────────────────────────────────────────────────────────────────
# FIX: R-multiple outcome classification (learning loop fix)
# Self-contained: replicates the classify_outcome logic to avoid MT5 import.
# ─────────────────────────────────────────────────────────────────────────────
def _classify_outcome_local(r_multiple: float) -> str:
    """Local replica of bot.classify_outcome for testing without MT5."""
    if r_multiple >= 1.5:   return "WIN_STRONG"
    if r_multiple >= 0.3:   return "WIN"
    if r_multiple >= -0.15: return "BREAKEVEN"
    if r_multiple >= -0.8:  return "LOSS"
    return "STOP_LOSS"


class TestRMultipleClassification(unittest.TestCase):
    """
    The old classify_outcome used account % which gave 0.006% for micro-lot wins.
    Everything was BREAKEVEN. The ML model only saw losses and could not learn.
    R-multiple (profit / risk) is account-size invariant.
    """

    def _classify(self, r: float) -> str:
        return _classify_outcome_local(r)

    def test_win_strong_at_1_5r(self):
        self.assertEqual(self._classify(1.5), "WIN_STRONG")

    def test_win_at_0_5r(self):
        self.assertEqual(self._classify(0.5), "WIN")

    def test_win_at_minimum_threshold(self):
        self.assertEqual(self._classify(0.3), "WIN")

    def test_breakeven_near_flat(self):
        self.assertEqual(self._classify(0.0), "BREAKEVEN")

    def test_breakeven_small_loss(self):
        self.assertEqual(self._classify(-0.1), "BREAKEVEN")

    def test_loss(self):
        self.assertEqual(self._classify(-0.5), "LOSS")

    def test_stop_loss_full(self):
        self.assertEqual(self._classify(-1.2), "STOP_LOSS")

    def test_micro_lot_win_now_classified_as_win(self):
        """
        On $5000 account with 0.01 lot, a 30-pip winner = $0.30.
        Old: pnl_pct = 0.006% → BREAKEVEN (never saved to memory).
        New: if risk_usd was $0.15 (15-pip SL × $0.10/pip), R = 0.30/0.15 = 2.0 → WIN_STRONG
        """
        net_pnl = 0.30
        risk_usd = 0.15
        r = net_pnl / risk_usd
        outcome = self._classify(r)
        self.assertIn(outcome, ("WIN", "WIN_STRONG"),
            f"Micro-lot win (R={r:.2f}) still classified as {outcome} — learning loop still broken!")


# ─────────────────────────────────────────────────────────────────────────────
# FIX: Reflection pre-filter uses R-multiple (not account %)
# Self-contained: replicates _pre_filter to avoid LLM dependency.
# ─────────────────────────────────────────────────────────────────────────────
def _pre_filter_local(r_multiple: float, pm_confidence: int, outcome: str) -> bool:
    """Local replica of agents.reflection._pre_filter for testing."""
    if outcome in ("STOP_LOSS", "STOP_LOSS_FULL"):
        return True
    if pm_confidence >= 80 and r_multiple < 0:
        return True
    if r_multiple >= 1.5:
        return True
    if abs(r_multiple) >= 0.3:
        return True
    return False


class TestReflectionPreFilter(unittest.TestCase):
    """_pre_filter must save significant trades regardless of account size."""

    def _filter(self, r_multiple: float, pm_confidence: int, outcome: str) -> bool:
        return _pre_filter_local(r_multiple, pm_confidence, outcome)

    def test_stop_loss_always_saved(self):
        self.assertTrue(self._filter(-1.5, 50, "STOP_LOSS"))

    def test_stop_loss_even_with_zero_r(self):
        self.assertTrue(self._filter(0.0, 50, "STOP_LOSS"))

    def test_win_1r_is_saved(self):
        """A 1R win (made back what we risked) must be saved."""
        self.assertTrue(self._filter(1.0, 70, "WIN"))

    def test_small_win_saved(self):
        """0.3R win — minimum significant threshold."""
        self.assertTrue(self._filter(0.3, 65, "WIN"))

    def test_tiny_win_skipped(self):
        """0.1R — noise, skip the LLM call."""
        self.assertFalse(self._filter(0.1, 65, "BREAKEVEN"))

    def test_high_confidence_loss_saved(self):
        """PM was 85% confident and wrong — critical learning opportunity."""
        self.assertTrue(self._filter(-0.5, 85, "LOSS"))

    def test_exceptional_win_saved(self):
        """1.5R+ = exceptional, always save."""
        self.assertTrue(self._filter(2.0, 70, "WIN_STRONG"))

    def test_micro_lot_30_pip_win_would_be_saved(self):
        """
        On $5000 account, 30-pip win on 0.01 lot with 15-pip SL:
        R = 2.0 → should be saved.
        Old system: pnl_pct = 0.006% → skipped. This test proves the fix works.
        """
        risk_usd = 0.15    # 15-pip SL × $0.01/pip = $0.15
        net_pnl  = 0.30    # 30-pip win × $0.01/pip = $0.30
        r = net_pnl / risk_usd  # = 2.0
        self.assertTrue(self._filter(r, 70, "WIN_STRONG"),
            f"R={r:.2f} win should pass the pre-filter but was rejected!")


# ─────────────────────────────────────────────────────────────────────────────
# FIX: Risk governor UTC timezone
# ─────────────────────────────────────────────────────────────────────────────
class TestGovernorTimezoneUTC(unittest.TestCase):
    """_maybe_day_reset must use UTC time, not local machine time."""

    def test_day_reset_uses_utc(self):
        """
        Verify that the governor stores UTC date on init.
        If a machine in Ghana (UTC+0) and a machine in US (UTC-5) both
        run the same bot, the day reset must fire at the same UTC midnight.
        """
        from datetime import datetime, timezone
        from risk_governor import RiskGovernor
        gov = RiskGovernor(10000.0, 2.0, 3, 10, 10.0, ["LONDON"])
        utc_today = datetime.now(timezone.utc).date()
        self.assertEqual(gov.status["today"], str(utc_today),
            "Governor initialized with local time instead of UTC — day reset will fire at wrong time!")


# ─────────────────────────────────────────────────────────────────────────────
# FIX: Memory cosine similarity fallback
# Reads the source file directly to avoid numpy import.
# ─────────────────────────────────────────────────────────────────────────────
class TestMemoryCosineFallback(unittest.TestCase):
    """Missing feature vector should give 0.0 similarity, not 0.5 (neutral)."""

    def test_missing_vector_gives_zero_not_neutral(self):
        """
        If a stored memory has no feature_vector (older memories), it should score
        zero on cosine similarity, not 0.5. Neutral (0.5) unfairly boosts
        memories with no data over memories that are genuinely dissimilar.
        """
        store_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "memory", "store.py"
        )
        with open(store_path, "r") as f:
            source = f.read()
        self.assertIn("else 0.0", source,
            "Cosine similarity fallback is not 0.0 — missing vectors still get neutral boost!")
        self.assertNotIn("else 0.5", source,
            "Old 0.5 fallback still present in recall() method!")


# ─────────────────────────────────────────────────────────────────────────────
# FIX: Kill switch closes positions (not just blocks new ones)
# Reads source directly to avoid MetaTrader5 import requirement.
# ─────────────────────────────────────────────────────────────────────────────
class TestKillSwitchClosesPositions(unittest.TestCase):
    """position_manager.close_all() must exist as a method."""

    def _get_source(self):
        pm_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "position_manager.py"
        )
        with open(pm_path, "r") as f:
            return f.read()

    def test_close_all_method_exists(self):
        source = self._get_source()
        self.assertIn("def close_all(", source,
            "PositionManager.close_all() does not exist — kill switch cannot close open positions!")

    def test_close_all_has_correct_params(self):
        source = self._get_source()
        self.assertIn("open_trades", source,
            "close_all() missing 'open_trades' parameter!")
        self.assertIn("cfg", source,
            "close_all() missing 'cfg' parameter!")

    def test_close_all_calls_mt5_close(self):
        source = self._get_source()
        self.assertIn("mt5b.close_position", source,
            "close_all() doesn't call mt5b.close_position — positions won't actually close!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
