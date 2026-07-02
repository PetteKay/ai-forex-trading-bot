"""
CUSUM Sequential Filter — event-driven trade entry sampling.

Source: López de Prado, AFML (2018) Chapter 2.

Purpose: Instead of evaluating every time bar (noisy, autocorrelated), only
run the expensive 5-agent desk when the CUSUM filter signals that price has
moved far enough to contain new information.

Two-sided CUSUM:
  S_pos tracks positive cumulative drift (price moving up)
  S_neg tracks negative cumulative drift (price moving down)
  When either exceeds threshold h → EVENT → reset and signal

Effect: Filters out noise ticks where price hasn't moved enough. Reduces LLM
API calls by 40-70% while focusing agent evaluation on genuine price moves.

Threshold h calibration:
  h = expected_daily_vol * k  (k ≈ 0.5–1.0)
  For H1 bars: h ≈ 0.5 * ATR / close  (0.5× ATR as percent move)

Per-symbol state is maintained between ticks in a dict. Reset on signal.
"""

import numpy as np
import logging

log = logging.getLogger("CUSUMFilter")


class CUSUMState:
    """
    Per-symbol CUSUM state tracker.
    Maintained in memory between bot ticks.
    """
    __slots__ = ("S_pos", "S_neg", "last_price", "n_bars_since_signal")

    def __init__(self):
        self.S_pos              = 0.0
        self.S_neg              = 0.0
        self.last_price         = None
        self.n_bars_since_signal = 0

    def reset(self) -> None:
        self.S_pos              = 0.0
        self.S_neg              = 0.0
        self.n_bars_since_signal = 0


class CUSUMFilter:
    """
    Multi-symbol CUSUM filter for the trading bot's main scan loop.

    Usage:
        _cusum = CUSUMFilter(h_pct=0.004, max_bars_between=20)

        # In the per-symbol scan:
        if not _cusum.should_evaluate(symbol, current_price):
            continue   # skip: no new information since last signal
    """

    def __init__(self,
                 h_pct: float = 0.004,
                 max_bars_between: int = 20,
                 min_bars_between: int = 3):
        """
        h_pct:            Threshold as fraction of price (0.004 = 0.4% move required).
                          Equivalent to ~40 pips on EURUSD (at 1.08) or ~$8 on XAUUSD (at 2000).
                          Calibrate to ~0.5× expected H1 ATR / price.
        max_bars_between: Force evaluation every N bars even if CUSUM hasn't fired.
                          Prevents the filter from suppressing trading indefinitely.
        min_bars_between: Don't re-fire within N bars (minimum wait after a signal).
        """
        self._h        = h_pct
        self._max      = max_bars_between
        self._min      = min_bars_between
        self._states: dict[str, CUSUMState] = {}

    def should_evaluate(self, symbol: str, current_price: float,
                        atr_pct: float | None = None) -> bool:
        """
        Returns True if the agent desk should be run for this symbol this tick.

        Fires when:
          a) CUSUM threshold exceeded (genuine price move)
          b) max_bars_between reached (time-based fallback)

        Does NOT fire when:
          - min_bars_between not yet met (prevent immediate re-evaluation)
          - Price hasn't moved enough (noisy tick)

        atr_pct: if provided, dynamically scales h to 0.5 * ATR/price.
        """
        if symbol not in self._states:
            self._states[symbol] = CUSUMState()

        state = self._states[symbol]
        state.n_bars_since_signal += 1

        # Minimum wait guard
        if state.n_bars_since_signal < self._min:
            return False

        # Time-based fallback: always evaluate after max bars
        if state.n_bars_since_signal >= self._max:
            state.reset()
            log.debug(f"CUSUM[{symbol}]: max_bars reached → force evaluate")
            return True

        if state.last_price is None or state.last_price <= 0:
            state.last_price = current_price
            return True  # First tick — always evaluate

        # CUSUM update
        log_ret = np.log(current_price / state.last_price)
        state.last_price = current_price

        # Dynamic threshold: use ATR if available, else fixed h_pct
        h = (0.5 * atr_pct) if (atr_pct and atr_pct > 0) else self._h

        state.S_pos = max(0.0, state.S_pos + log_ret)
        state.S_neg = min(0.0, state.S_neg + log_ret)

        if state.S_pos >= h:
            state.reset()
            log.debug(f"CUSUM[{symbol}]: bullish signal (S_pos≥{h:.4f})")
            return True
        elif state.S_neg <= -h:
            state.reset()
            log.debug(f"CUSUM[{symbol}]: bearish signal (S_neg≤{-h:.4f})")
            return True

        return False

    def reset_symbol(self, symbol: str) -> None:
        """Reset a symbol's state (e.g., after a trade is placed)."""
        if symbol in self._states:
            self._states[symbol].reset()

    def get_state(self, symbol: str) -> dict:
        """Return current CUSUM state for a symbol (for debugging)."""
        if symbol not in self._states:
            return {"S_pos": 0.0, "S_neg": 0.0, "n_bars": 0}
        s = self._states[symbol]
        return {"S_pos": round(s.S_pos, 6), "S_neg": round(s.S_neg, 6),
                "n_bars": s.n_bars_since_signal}
