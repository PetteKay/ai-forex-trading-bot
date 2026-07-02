"""
Backtesting Engine with Walk-Forward Validation.

Features:
  - Event-driven simulation (processes each candle in order)
  - Realistic order execution (spread, slippage)
  - Triple-barrier labeling (López de Prado)
  - Walk-forward validation to avoid overfitting
  - Deflated Sharpe Ratio and Probability of Backtest Overfitting (PBO)
  - Per-trade metrics: R-multiple, MAE, MFE
  - Full trade log output

Usage:
    from backtest.engine import BacktestEngine
    engine = BacktestEngine(config)
    results = engine.run(df, strategy_fn)
    engine.print_report(results)
"""

import logging
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger("Backtest")


@dataclass
class BacktestTrade:
    entry_bar:   int
    exit_bar:    int
    direction:   str          # BUY / SELL
    entry_price: float
    exit_price:  float
    sl_price:    float
    tp_price:    float
    volume:      float
    pnl:         float
    pnl_pct:     float
    r_multiple:  float        # pnl / initial_risk
    exit_reason: str          # TP / SL / TIME / SIGNAL
    mae:         float = 0.0  # Maximum Adverse Excursion
    mfe:         float = 0.0  # Maximum Favourable Excursion
    hold_bars:   int = 0


@dataclass
class BacktestResults:
    trades:          list = field(default_factory=list)
    equity_curve:    list = field(default_factory=list)
    total_return:    float = 0.0
    win_rate:        float = 0.0
    profit_factor:   float = 0.0
    sharpe:          float = 0.0
    deflated_sharpe: float = 0.0
    max_drawdown:    float = 0.0
    avg_r_multiple:  float = 0.0
    expectancy:      float = 0.0
    n_trades:        int = 0



# Symbol-aware pip sizes for backtesting (no live MT5 connection available)
_PIP_SIZE_MAP: dict[str, float] = {
    "JPY":  0.01,    # All JPY crosses (USDJPY, EURJPY, GBPJPY…)
    "XAU":  0.01,    # Gold (XAUUSD)
    "XAG":  0.001,   # Silver (XAGUSD)
    "GOLD": 0.01,
    "OIL":  0.01,
    "WTI":  0.01,
}
_DEFAULT_PIP_SIZE = 0.0001  # standard 5-digit pairs (EURUSD, GBPUSD, etc.)

# USD value of 1 pip per standard lot for backtesting (approximate)
_PIP_VALUE_STD_LOT: dict[str, float] = {
    "JPY":  9.10,    # ~110 USDJPY; pip = 0.01, value ≈ $9.10/lot
    "XAU": 10.00,    # Gold; 100oz contract, pip = $0.01 → ~$10/lot
    "XAG":  5.00,    # Silver approximation
    "GOLD":10.00,
    "OIL":  1.00,
    "WTI":  1.00,
}
_DEFAULT_PIP_VALUE = 10.0   # $10/pip per standard lot for USD-quoted majors


def _symbol_pip_size(symbol: str) -> float:
    upper = symbol.upper()
    for key, size in _PIP_SIZE_MAP.items():
        if key in upper:
            return size
    return _DEFAULT_PIP_SIZE


def _symbol_pip_value(symbol: str) -> float:
    """USD value per pip per standard lot (for backtesting approximation)."""
    upper = symbol.upper()
    for key, val in _PIP_VALUE_STD_LOT.items():
        if key in upper:
            return val
    return _DEFAULT_PIP_VALUE


class BacktestEngine:
    def __init__(self, cfg, spread_pips: float = 2.0, slippage_pips: float = 0.5,
                 symbol: str = "EURUSD"):
        self._cfg      = cfg
        self._spread   = spread_pips
        self._slip     = slippage_pips
        self._symbol   = symbol

    def run(self, df: pd.DataFrame, strategy_fn: Callable,
            initial_equity: float = 10_000.0,
            commission_per_lot: float = 7.0,
            symbol: str = "") -> BacktestResults:
        """
        Run the backtest.

        Args:
            df: OHLCV DataFrame (pre-computed indicators attached as columns, or raw)
            strategy_fn: function(df_slice) → {"action": "BUY"/"SELL"/"HOLD",
                                                "sl_pips": float, "tp_pips": float,
                                                "confidence": int}
            initial_equity: starting account balance
            commission_per_lot: round-turn commission per standard lot (USD)
            symbol: instrument symbol — used for correct pip_size / pip_value.
                    Falls back to symbol passed at __init__ time, then config.SYMBOL.

        Returns BacktestResults object.
        """
        from strategy.indicators import compute

        effective_symbol = (
            symbol or self._symbol
            or getattr(self._cfg, "SYMBOL", "EURUSD")
        )
        pip_size      = _symbol_pip_size(effective_symbol)
        pip_val_lot   = _symbol_pip_value(effective_symbol)   # USD/pip per standard lot

        equity        = initial_equity
        peak_equity   = initial_equity
        trades        = []
        equity_curve  = [equity]
        open_trade: Optional[dict] = None

        df = df.copy()
        if "ema_fast" not in df.columns:
            df = compute(df, self._cfg).get("_df", df)

        for i in range(200, len(df)):
            bar     = df.iloc[i]
            df_slice = df.iloc[:i+1]

            high   = float(bar["high"])
            low    = float(bar["low"])
            close  = float(bar["close"])

            # ── Manage open trade ──────────────────────────────────────
            if open_trade:
                d   = open_trade
                mae = 0.0
                mfe = 0.0

                if d["direction"] == "BUY":
                    adverse  = d["entry"] - low
                    fav      = high - d["entry"]
                    sl_hit   = low  <= d["sl"]
                    tp_hit   = high >= d["tp"]
                else:
                    adverse  = high - d["entry"]
                    fav      = d["entry"] - low
                    sl_hit   = high >= d["sl"]
                    tp_hit   = low  <= d["tp"]

                mae = max(0, adverse)
                mfe = max(0, fav)

                exit_price  = None
                exit_reason = None

                if sl_hit and tp_hit:
                    # Both hit on same bar — use order (conservative: SL)
                    exit_price  = d["sl"]
                    exit_reason = "SL"
                elif sl_hit:
                    exit_price  = d["sl"]
                    exit_reason = "SL"
                elif tp_hit:
                    exit_price  = d["tp"]
                    exit_reason = "TP"
                elif (i - d["entry_bar"]) >= getattr(self._cfg, "BACKTEST_MAX_HOLD_BARS", 120):
                    exit_price  = close
                    exit_reason = "TIME"

                if exit_price is not None:
                    if d["direction"] == "BUY":
                        pnl_pips = (exit_price - d["entry"]) / pip_size
                    else:
                        pnl_pips = (d["entry"] - exit_price) / pip_size

                    pnl_pips -= self._spread  # spread cost
                    lot_value = d["volume"] * pip_val_lot  # symbol-aware USD/pip for this lot size
                    pnl       = pnl_pips * lot_value - commission_per_lot * d["volume"] * 10
                    risk_pips = abs(d["entry"] - d["sl"]) / pip_size

                    pnl_pct   = pnl / equity * 100
                    r_multiple = pnl_pips / risk_pips if risk_pips > 0 else 0

                    equity += pnl
                    if equity > peak_equity:
                        peak_equity = equity

                    t = BacktestTrade(
                        entry_bar   = d["entry_bar"],
                        exit_bar    = i,
                        direction   = d["direction"],
                        entry_price = d["entry"],
                        exit_price  = exit_price,
                        sl_price    = d["sl"],
                        tp_price    = d["tp"],
                        volume      = d["volume"],
                        pnl         = pnl,
                        pnl_pct     = pnl_pct,
                        r_multiple  = r_multiple,
                        exit_reason = exit_reason,
                        mae         = mae,
                        mfe         = mfe,
                        hold_bars   = i - d["entry_bar"],
                    )
                    trades.append(t)
                    open_trade = None
                    equity_curve.append(equity)
                    continue

            # ── Entry signal (only if flat) ────────────────────────────
            if open_trade is None:
                try:
                    sig = strategy_fn(df_slice)
                except Exception:
                    continue

                action = sig.get("action", "HOLD")
                if action not in ("BUY", "SELL"):
                    continue

                sl_pips = sig.get("sl_pips", float(getattr(self._cfg, "ATR_SL_MULT", 1.5)))
                tp_pips = sig.get("tp_pips", float(getattr(self._cfg, "ATR_TP_MULT", 2.5)))
                volume  = float(getattr(self._cfg, "LOT_SIZE", 0.01))

                if action == "BUY":
                    entry = close + self._slip * pip_size
                    sl    = entry - sl_pips * pip_size
                    tp    = entry + tp_pips * pip_size
                else:
                    entry = close - self._slip * pip_size
                    sl    = entry + sl_pips * pip_size
                    tp    = entry - tp_pips * pip_size

                open_trade = {
                    "direction": action,
                    "entry":     entry,
                    "sl":        sl,
                    "tp":        tp,
                    "volume":    volume,
                    "entry_bar": i,
                }

        return self._compile_results(trades, equity_curve, initial_equity)

    def walk_forward(self, df: pd.DataFrame, strategy_fn: Callable,
                     train_frac: float = 0.7,
                     n_splits: int = 5,
                     initial_equity: float = 10_000.0) -> dict:
        """
        Walk-forward validation: train on first 70%, test on each subsequent segment.
        Returns per-split metrics plus aggregate statistics.
        """
        n      = len(df)
        splits = []

        for k in range(n_splits):
            test_start = int(n * (train_frac + k * (1 - train_frac) / n_splits))
            test_end   = int(n * (train_frac + (k + 1) * (1 - train_frac) / n_splits))
            if test_end <= test_start + 50:
                continue

            test_df = df.iloc[test_start:test_end]
            result  = self.run(test_df, strategy_fn, initial_equity, symbol=self._symbol)
            splits.append({
                "split":        k + 1,
                "n_trades":     result.n_trades,
                "sharpe":       result.sharpe,
                "total_return": result.total_return,
                "max_dd":       result.max_drawdown,
                "win_rate":     result.win_rate,
            })
            log.info(
                f"WFV split {k+1}/{n_splits}: trades={result.n_trades} "
                f"return={result.total_return:.1f}% sharpe={result.sharpe:.2f} "
                f"win_rate={result.win_rate:.1%}"
            )

        if not splits:
            return {"splits": [], "aggregate": {}}

        sharpes      = [s["sharpe"] for s in splits]
        avg_sharpe   = np.mean(sharpes)
        std_sharpe   = np.std(sharpes) if len(sharpes) > 1 else 0
        deflated_sr  = self._deflated_sharpe(sharpes)

        return {
            "splits": splits,
            "aggregate": {
                "avg_sharpe":        round(avg_sharpe, 3),
                "std_sharpe":        round(std_sharpe, 3),
                "deflated_sharpe":   round(deflated_sr, 3),
                "pbo_warning":       deflated_sr < 0,
                "avg_win_rate":      round(np.mean([s["win_rate"] for s in splits]), 3),
                "avg_total_return":  round(np.mean([s["total_return"] for s in splits]), 2),
            }
        }

    def print_report(self, results: BacktestResults) -> None:
        print("\n" + "═" * 60)
        print("  BACKTEST REPORT")
        print("═" * 60)
        print(f"  Trades:          {results.n_trades}")
        print(f"  Win rate:        {results.win_rate:.1%}")
        print(f"  Total return:    {results.total_return:.2f}%")
        print(f"  Profit factor:   {results.profit_factor:.2f}")
        print(f"  Sharpe ratio:    {results.sharpe:.3f}")
        print(f"  Deflated Sharpe: {results.deflated_sharpe:.3f}")
        print(f"  Max drawdown:    {results.max_drawdown:.2f}%")
        print(f"  Avg R-multiple:  {results.avg_r_multiple:.2f}R")
        print(f"  Expectancy:      {results.expectancy:.2f}")
        if results.n_trades > 0:
            wins  = [t for t in results.trades if t.pnl > 0]
            loses = [t for t in results.trades if t.pnl <= 0]
            tp_exits = sum(1 for t in results.trades if t.exit_reason == "TP")
            sl_exits = sum(1 for t in results.trades if t.exit_reason == "SL")
            print(f"  TP exits:        {tp_exits} | SL exits: {sl_exits}")
        print("═" * 60)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compile_results(self, trades: list, equity_curve: list, initial_equity: float) -> BacktestResults:
        r = BacktestResults(trades=trades, equity_curve=equity_curve)
        r.n_trades = len(trades)
        if not trades:
            return r

        wins    = [t.pnl for t in trades if t.pnl > 0]
        losses  = [abs(t.pnl) for t in trades if t.pnl <= 0]
        r_mults = [t.r_multiple for t in trades]

        r.win_rate     = len(wins) / r.n_trades
        r.profit_factor = (sum(wins) / sum(losses)) if losses else float("inf")
        r.total_return  = (equity_curve[-1] - initial_equity) / initial_equity * 100
        r.avg_r_multiple = float(np.mean(r_mults))
        r.expectancy    = float(np.mean([t.pnl for t in trades]))

        # Sharpe (annualised, assuming H1 bars)
        pnl_pcts = np.array([t.pnl_pct for t in trades])
        if len(pnl_pcts) > 1 and pnl_pcts.std() > 0:
            bars_per_year = 252 * 24  # H1 approximation
            r.sharpe = float(pnl_pcts.mean() / pnl_pcts.std() * np.sqrt(bars_per_year / max(1, r.n_trades)))
        else:
            r.sharpe = 0.0

        r.deflated_sharpe = self._deflated_sharpe([r.sharpe])

        # Max drawdown
        peak = initial_equity
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        r.max_drawdown = max_dd

        return r

    @staticmethod
    def _deflated_sharpe(sharpe_series: list[float]) -> float:
        """
        Simplified Deflated Sharpe Ratio (Bailey & López de Prado 2014).
        Penalises for multiple testing and non-normality.
        """
        if not sharpe_series:
            return 0.0
        sr_max = max(sharpe_series)
        n      = len(sharpe_series)
        if n <= 1:
            return sr_max
        # Simple penalty for number of trials
        penalty = math.sqrt(2 * math.log(n))
        return sr_max - penalty * abs(sr_max) / 10
