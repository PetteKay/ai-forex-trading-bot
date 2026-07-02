"""
MT5 Historical Data Trainer — bootstraps the ML meta-label gate.

WHY THIS EXISTS
--------------
The ML gate needs 30+ trade memories before it activates. Without pre-seeding,
the bot trades on heuristics for weeks. This script uses MT5's own bar history
to generate hundreds of training samples instantly.

ZERO DISTRIBUTION SHIFT
-----------------------
It uses the exact same data source (MT5), the exact same indicator pipeline
(strategy.indicators.compute + feature_vector), and the exact same
ATR_SL_MULT/ATR_TP_MULT from config as the live bot. Feature vectors computed
here are on the identical price distribution the model will see at inference.
That's not possible with yfinance or Kaggle CSVs.

NO LLM CALLS
------------
Samples are written directly to the memory DB. Saves ~$0.005–0.03 per sample.
500 samples ≈ $0 vs $2.50–$15 through the reflection agent.

USAGE
-----
  python mt5_historical_trainer.py                     # 5000 H1 bars, config.SYMBOL
  python mt5_historical_trainer.py --bars 10000        # more history (~2 years H1)
  python mt5_historical_trainer.py --symbol XAUUSD     # gold
  python mt5_historical_trainer.py --max-samples 400   # smaller memory footprint
"""

import argparse
import json
import logging
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

import config
import mt5_bridge as mt5b
from memory.store import TradeMemoryStore
from strategy.indicators import compute, feature_vector, market_fingerprint
from strategy.regime import classify as classify_regime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("HistoricalTrainer")

# Bars needed for EMA200, MACD, Bollinger, ADX, Hurst to be fully warmed up
_WARMUP = 260

# Usefulness scores by outcome — all above the 0.55 save threshold
_USEFULNESS = {
    "WIN_STRONG": 0.72,
    "WIN":        0.65,
    "BREAKEVEN":  0.60,
    "LOSS":       0.65,
    "STOP_LOSS":  0.72,   # highest — learning what kills trades is most valuable
}


# ── Direction signal ────────────────────────────────────────────────────────

def _get_direction(ctx: dict, regime: str) -> str | None:
    """
    Return 'BUY', 'SELL', or None (no tradeable signal).

    Uses the same multi-condition confluence the live fund manager approves:
    EMA alignment + MACD + RSI not at extremes + no squeeze + no hostile regime.
    Strict on purpose — bad signals make worse training data than fewer signals.
    """
    if regime in ("VOLATILE", "RANGING", "UNKNOWN"):
        return None

    rsi       = ctx.get("rsi", 50)
    fast_bull = ctx.get("fast_above_slow", False)
    abv_trend = ctx.get("price_above_trend", False)
    macd_bull = ctx.get("macd_positive", False)
    squeeze   = ctx.get("bb_squeeze", False)
    struct    = ctx.get("market_structure", "SIDEWAYS")

    if squeeze:
        return None

    # BUY: full EMA stack bullish, MACD positive, RSI not overbought
    if (fast_bull and abv_trend and macd_bull
            and 40 <= rsi <= 70
            and struct in ("UPTREND", "SIDEWAYS")):
        return "BUY"

    # SELL: full EMA stack bearish, MACD negative, RSI not oversold
    if (not fast_bull and not abv_trend and not macd_bull
            and 30 <= rsi <= 60
            and struct in ("DOWNTREND", "SIDEWAYS")):
        return "SELL"

    return None


# ── Triple-barrier labeler ──────────────────────────────────────────────────

def _triple_barrier(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    atr: float,
    cfg,
) -> tuple[str, float, int]:
    """
    Label a trade using the triple-barrier method (same SL/TP ratios as live bot).

    Barriers:
      TP: entry ± ATR × ATR_TP_MULT
      SL: entry ∓ ATR × ATR_SL_MULT
      Time: MAX_HOLD_HOURS bars forward

    Returns (outcome, r_multiple, bars_held).
    r_multiple = tp_dist / sl_dist on a winner, -1.0 on a full stop, proportional otherwise.
    """
    entry_price = float(df.iloc[entry_idx]["close"])
    sl_dist = atr * cfg.ATR_SL_MULT
    tp_dist = atr * cfg.ATR_TP_MULT
    tp_r    = tp_dist / sl_dist       # e.g. 2.5 / 1.5 = 1.667

    if sl_dist <= 0:
        return "BREAKEVEN", 0.0, 1

    max_bars = min(cfg.MAX_HOLD_HOURS, len(df) - entry_idx - 1)

    for j in range(1, max_bars + 1):
        bar  = df.iloc[entry_idx + j]
        high = float(bar["high"])
        low  = float(bar["low"])

        if direction == "BUY":
            if high >= entry_price + tp_dist:
                return ("WIN_STRONG" if tp_r >= 1.5 else "WIN"), tp_r, j
            if low <= entry_price - sl_dist:
                return "STOP_LOSS", -1.0, j
        else:                                      # SELL
            if low <= entry_price - tp_dist:
                return ("WIN_STRONG" if tp_r >= 1.5 else "WIN"), tp_r, j
            if high >= entry_price + sl_dist:
                return "STOP_LOSS", -1.0, j

    # Time barrier — compute actual floating P&L at exit
    j = max_bars
    exit_price = float(df.iloc[entry_idx + j]["close"])
    raw_pnl    = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
    r          = raw_pnl / sl_dist

    if r >= 1.5:    return "WIN_STRONG", r,    j
    if r >= 0.3:    return "WIN",        r,    j
    if r >= -0.15:  return "BREAKEVEN",  r,    j
    if r >= -0.8:   return "LOSS",       r,    j
    return "STOP_LOSS", r, j


# ── Lesson generator ────────────────────────────────────────────────────────

def _make_lesson(direction: str, outcome: str, ctx: dict, r_multiple: float, bars_held: int) -> str:
    struct = ctx.get("market_structure", "UNKNOWN")
    rsi_st = ctx.get("rsi_state", "NEUTRAL")
    htf_b  = ctx.get("htf_bias", "NEUTRAL")
    rsi    = ctx.get("rsi", 50)

    if outcome == "WIN_STRONG":
        return (f"{direction} in {struct}: full {r_multiple:.1f}R win in {bars_held}h. "
                f"EMA+MACD+regime confluence. HTF={htf_b}, RSI={rsi:.0f}. "
                f"This setup pays when all three filters align.")
    if outcome == "WIN":
        return (f"{direction} in {struct}: modest {r_multiple:.1f}R win in {bars_held}h. "
                f"RSI={rsi:.0f} ({rsi_st}). Valid setup but fell short of full TP — "
                f"consider tighter partial-close target.")
    if outcome == "BREAKEVEN":
        return (f"{direction} in {struct}: flat after {bars_held}h (RSI={rsi:.0f}, HTF={htf_b}). "
                f"EMA+MACD aligned but no follow-through — likely regime conflict or "
                f"consolidation phase. Avoid unless HTF confirms strongly.")
    if outcome == "LOSS":
        return (f"{direction} in {struct}: partial loss {r_multiple:.1f}R in {bars_held}h. "
                f"RSI={rsi:.0f}, HTF={htf_b}. Setup started but reversed — "
                f"check ATR expansion before entry.")
    # STOP_LOSS
    return (f"{direction} STOPPED in {struct} ({bars_held}h, RSI={rsi:.0f}, HTF={htf_b}). "
            f"EMA+MACD confluence failed. Market reversed immediately after signal. "
            f"If {rsi_st} and {struct} conflict with HTF direction, skip the trade.")


# ── Direct DB write (bypasses LLM reflection cost) ─────────────────────────

def _write_sample(db_path: str, s: dict) -> bool:
    fv_json = json.dumps(s["feature_vector"]) if s.get("feature_vector") else None
    mc      = {k: v for k, v in s.get("market_context", {}).items() if k != "_df"}

    try:
        with sqlite3.connect(db_path) as con:
            con.execute("""
                INSERT INTO memories
                  (timestamp, symbol, direction, entry_price, exit_price,
                   pnl_pct, r_multiple, outcome, fingerprint, feature_vector,
                   market_context, agent_reasoning, lesson, usefulness_score,
                   pattern_name, category, times_recalled, regime)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)
            """, (
                s["timestamp"],
                s["symbol"],
                s["direction"],
                s["entry_price"],
                s["exit_price"],
                s["pnl_pct"],
                s["r_multiple"],
                s["outcome"],
                s["fingerprint"],
                fv_json,
                json.dumps(mc),
                "Historical: EMA+MACD+regime confluence signal (no LLM)",
                s["lesson"],
                s["usefulness_score"],
                s["fingerprint"],      # pattern_name = fingerprint
                "ENTRY_TIMING",
                s.get("regime", ""),   # enables regime-conditional meta-label models
            ))
        return True
    except Exception as e:
        log.error(f"DB write error: {e}")
        return False


# ── Main ────────────────────────────────────────────────────────────────────

def run(args):
    symbol    = args.symbol or config.SYMBOL
    timeframe = args.timeframe or config.TIMEFRAME
    n_bars    = args.bars
    max_samp  = args.max_samples

    log.info("=" * 65)
    log.info("MT5 Historical Data Trainer")
    log.info(f"Symbol: {symbol}  TF: {timeframe}  Bars: {n_bars}  Max samples: {max_samp}")
    log.info("=" * 65)

    if not mt5b.connect():
        log.error("MT5 terminal must be open and logged in.")
        sys.exit(1)

    symbol = mt5b.resolve_symbol(symbol)

    # ── Fetch primary-timeframe bars ────────────────────────────────────────
    log.info(f"Fetching {n_bars} {timeframe} bars for {symbol}...")
    try:
        df = mt5b.get_candles(symbol, timeframe, n_bars)
    except Exception as e:
        log.error(f"Failed to fetch bars: {e}")
        mt5b.disconnect()
        sys.exit(1)

    log.info(f"Got {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    if len(df) < _WARMUP + 60:
        log.error(f"Need at least {_WARMUP + 60} bars. Got {len(df)}.")
        mt5b.disconnect()
        sys.exit(1)

    # ── Fetch HTF bars ──────────────────────────────────────────────────────
    htf_tf = getattr(config, "HTF_TIMEFRAME", "H4")
    htf_n  = max(600, n_bars // 4 + 200)
    log.info(f"Fetching {htf_n} {htf_tf} bars for HTF bias...")
    htf_df: pd.DataFrame | None = None
    try:
        htf_df = mt5b.get_candles(symbol, htf_tf, htf_n)
        log.info(f"HTF: {len(htf_df)} bars")
    except Exception as e:
        log.warning(f"HTF unavailable ({e}) — HTF bias = NEUTRAL for all samples")

    # ── Ensure memory DB schema exists ──────────────────────────────────────
    store = TradeMemoryStore(
        config.MEMORY_DB_PATH,
        config.MEMORY_USEFULNESS_THRESHOLD,
        config.MEMORY_MAX_STORED,
    )
    log.info(f"Existing memory: {store.stats()}")

    # ── Scan bars for tradeable signals ────────────────────────────────────
    # Leave MAX_HOLD_HOURS bars at the end — no forward data for labeling
    end_idx  = len(df) - config.MAX_HOLD_HOURS - 1
    signals  = []
    skipped_regime = 0
    skipped_no_signal = 0

    log.info(f"Scanning bars {_WARMUP}–{end_idx} ({end_idx - _WARMUP} bars)...")

    for i in range(_WARMUP, end_idx):
        h1_slice = df.iloc[: i + 1]

        # HTF slice up to the current bar's timestamp
        if htf_df is not None:
            bar_time  = df.index[i]
            htf_slice = htf_df[htf_df.index <= bar_time]
            htf_arg   = htf_slice if len(htf_slice) >= 5 else None
        else:
            htf_arg = None

        try:
            ctx = compute(h1_slice, config, htf_df=htf_arg)
        except Exception:
            continue
        if not ctx:
            continue

        # Regime — same classifier as live bot
        try:
            lookback_df = h1_slice.iloc[-min(len(h1_slice), 500):]
            regime_info = classify_regime(lookback_df, ctx.get("atr_percentile", 50))
            regime      = regime_info.get("regime", "UNKNOWN")
        except Exception:
            regime = "UNKNOWN"

        direction = _get_direction(ctx, regime)
        if direction is None:
            if regime in ("VOLATILE", "RANGING", "UNKNOWN"):
                skipped_regime += 1
            else:
                skipped_no_signal += 1
            continue

        atr = ctx.get("atr", 0.0)
        if atr <= 0:
            continue

        outcome, r_multiple, bars_held = _triple_barrier(df, i, direction, atr, config)

        fv = feature_vector(ctx)
        fp = market_fingerprint(ctx)

        bar_ts      = df.index[i]
        entry_price = float(df.iloc[i]["close"])
        exit_price  = float(df.iloc[i + bars_held]["close"])
        # pnl_pct assumes 1% risk per trade (display only — ML uses r_multiple)
        pnl_pct     = round(r_multiple * 1.0, 4)

        signals.append({
            "timestamp":       bar_ts.isoformat(),
            "symbol":          symbol,
            "regime":          regime,
            "direction":       direction,
            "entry_price":     entry_price,
            "exit_price":      exit_price,
            "pnl_pct":         pnl_pct,
            "r_multiple":      round(r_multiple, 4),
            "outcome":         outcome,
            "fingerprint":     fp,
            "feature_vector":  fv,
            "market_context":  {k: v for k, v in ctx.items() if k != "_df"},
            "lesson":          _make_lesson(direction, outcome, ctx, r_multiple, bars_held),
            "usefulness_score": _USEFULNESS.get(outcome, 0.65),
        })

        if len(signals) % 200 == 0 and len(signals) > 0:
            pct = (i - _WARMUP) / (end_idx - _WARMUP) * 100
            log.info(f"  {pct:.0f}% done — {len(signals)} signals found")

    log.info(f"\nScan complete:")
    log.info(f"  Signals found:        {len(signals)}")
    log.info(f"  Skipped (regime):     {skipped_regime}")
    log.info(f"  Skipped (no signal):  {skipped_no_signal}")

    if not signals:
        log.warning("No signals generated. Try --bars 8000 or check the symbol.")
        mt5b.disconnect()
        return

    # ── Outcome distribution ────────────────────────────────────────────────
    by_outcome: dict[str, list] = defaultdict(list)
    for s in signals:
        by_outcome[s["outcome"]].append(s)

    log.info("\nOutcome distribution (raw):")
    for outcome, items in sorted(by_outcome.items()):
        log.info(f"  {outcome:15s}: {len(items):5d}  ({100 * len(items) / len(signals):.1f}%)")

    win_rate = sum(
        1 for s in signals if s["outcome"] in ("WIN", "WIN_STRONG")
    ) / len(signals)
    log.info(f"  Raw win rate: {win_rate:.1%}")
    if win_rate < 0.35:
        log.warning(
            "Win rate below 35% in history — strategy has negative historical expectancy "
            "on this symbol/timeframe. Consider tuning ATR multipliers or ALLOWED_REGIMES."
        )

    # ── Stratified sample to fit within max_samples ─────────────────────────
    # Take proportional samples from each outcome class (time-diverse within class)
    if len(signals) > max_samp:
        log.info(f"\nStratified sampling: {max_samp} from {len(signals)}...")
        per_class = max(1, max_samp // max(len(by_outcome), 1))
        selected  = []
        for outcome, items in by_outcome.items():
            # Evenly spaced (not purely random) for temporal diversity
            step = max(1, len(items) // per_class)
            selected.extend(items[::step][:per_class])
        # Fill any remaining quota from the leftover pool
        remaining = max_samp - len(selected)
        if remaining > 0:
            selected_set = set(id(s) for s in selected)
            pool = [s for s in signals if id(s) not in selected_set]
            extra = random.sample(pool, min(remaining, len(pool)))
            selected.extend(extra)
        random.shuffle(selected)
        signals = selected[:max_samp]
        log.info(f"  After sampling: {len(signals)}")

    # ── Write to memory DB ──────────────────────────────────────────────────
    log.info(f"\nWriting {len(signals)} samples to {config.MEMORY_DB_PATH}...")
    saved = failed = 0
    for s in signals:
        if _write_sample(config.MEMORY_DB_PATH, s):
            saved += 1
        else:
            failed += 1

    # Prune if over capacity (same logic as live store)
    store._prune_if_needed()

    log.info("")
    log.info("=" * 65)
    log.info(f"Done: {saved} samples saved, {failed} failed")
    log.info(f"Final memory: {store.stats()}")
    log.info("")
    log.info("Next steps:")
    log.info("  1. python bot.py --demo    (ML gate activates on first tick)")
    log.info("  2. python bot.py           (live trading with pre-trained ML)")
    log.info("  The model retrains automatically every 20 new live memories.")
    log.info("=" * 65)

    mt5b.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed bot memory with MT5 historical bar data (no LLM, no external APIs)"
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=5000,
        help="Number of historical bars to process (default: 5000 ≈ 7 months H1)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Symbol (default: config.SYMBOL)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Timeframe (default: config.TIMEFRAME)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        dest="max_samples",
        help="Max training samples to save (default: 500)",
    )
    run(parser.parse_args())
