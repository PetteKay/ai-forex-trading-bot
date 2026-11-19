"""
New-candle gate.

Signals are computed from CLOSED candles, which only change when a new candle
closes (once an hour on H1). The bot ticks every 60 s, and with the CUSUM
fallback (CUSUM_MAX_BARS = 10 ticks) it re-ran the full agent desk about
7 times an hour on the same candle (measured) — the same question asked
again until the LLM's randomness gave a different answer.

The gate lets the agent desk run once per new closed candle per symbol.
"""


class BarGate:
    def __init__(self):
        self._evaluated: dict[str, object] = {}

    def already_evaluated(self, symbol: str, bar_time) -> bool:
        """True if the desk already ran for this symbol on this closed candle."""
        return self._evaluated.get(symbol) == bar_time

    def mark(self, symbol: str, bar_time) -> None:
        """Record that the desk ran on this candle."""
        self._evaluated[symbol] = bar_time

    def retry(self, symbol: str) -> None:
        """Allow another evaluation on the same candle (e.g. M15 timing said WAIT)."""
        self._evaluated.pop(symbol, None)
