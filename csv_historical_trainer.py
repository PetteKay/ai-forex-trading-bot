"""
CSV Historical Tick Trainer — bootstraps the ML meta-label gate from raw MT5
tick-export CSVs, without needing a live MT5 terminal connection.

WHY THIS EXISTS
----------------
mt5_historical_trainer.py needs a running, logged-in MT5 terminal to pull bar
history. When you already have exported tick data (MT5 Tools > History
Center > Export Ticks, or a vendor dump) this script resamples those ticks
into the same H1/H4 bar structure and feeds them through the *identical*
indicator / regime / triple-barrier pipeline (imported directly from
mt5_historical_trainer.py) so training samples stay consistent with that
script's output — same feature vectors, same labeling, same DB writer.

CSV FORMAT EXPECTED (MT5 tick export, no header)
--------------------------------------------------
  <DATE>,<TIME>,<BID>,<ASK>,<LAST>,<VOLUME>,<FLAGS>
  2026.06.01,03:00:00.817,1.16486,1.16487,0,1.79999995,0

Files are matched by name: <SYMBOL>_TICK_*.csv in the given directory.

USAGE
-----
  python csv_historical_trainer.py                     # all *_TICK_*.csv in cwd
  python csv_historical_trainer.py --dir data/ticks     # custom directory
  python csv_historical_trainer.py --symbol EURUSD      # just one file
  python csv_historical_trainer.py --max-samples 200    # per-symbol cap
"""

import argparse
import glob
import logging
import os
import random
import sys
from collections import defaultdict

import pandas as pd

import config
from memory.store import TradeMemoryStore
from strategy.indicators import compute, feature_vector, market_fingerprint
from strategy.regime import classify as classify_regime

from mt5_historical_trainer import (
    _WARMUP,
    _USEFULNESS,
    _get_direction,
    _triple_barrier,
    _make_lesson,
    _write_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("CSVHistoricalTrainer")

_RESAMPLE_RULE = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D",
}


# ── Tick loading / resampling ───────────────────────────────────────────────

def _load_ticks(path: str) -> pd.Series:
    """Load an MT5 tick-export CSV and return a mid-price Series indexed by datetime."""
    df = pd.read_csv(
        path,
        header=None,
        names=["date", "time", "bid", "ask", "last", "volume", "flags"],
        usecols=["date", "time", "bid", "ask"],
        dtype={"date": str, "time": str, "bid": float, "ask": float},
    )
    ts  = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S.%f")
    mid = (df["bid"] + df["ask"]) / 2.0
    mid.index = ts
    mid = mid.sort_index()
    mid.name = "mid"
    return mid


def _resample(mid: pd.Series, rule: str) -> pd.DataFrame:
    """Resample a tick mid-price series into OHLCV bars (volume = tick count)."""
    o = mid.resample(rule).ohlc()
    v = mid.resample(rule).size().rename("volume")
    bars = pd.concat([o, v], axis=1).dropna(subset=["open"])
    return bars


def _find_csv(directory: str, symbol: str) -> str | None:
    matches = glob.glob(os.path.join(directory, f"{symbol}_TICK_*.csv"))
    return matches[0] if matches else None


# ── Per-symbol scan (mirrors mt5_historical_trainer.run()'s bar loop) ──────

def _scan_symbol(symbol: str, df: pd.DataFrame, htf_df: pd.DataFrame) -> list[dict]:
    end_idx = len(df) - config.MAX_HOLD_HOURS - 1
    signals: list[dict] = []
    skipped_regime = 0
    skipped_no_signal = 0

    for i in range(_WARMUP, end_idx):
        h1_slice = df.iloc[: i + 1]
        bar_time = df.index[i]

        htf_slice = htf_df[htf_df.index <= bar_time]
        htf_arg   = htf_slice if len(htf_slice) >= 5 else None

        try:
            ctx = compute(h1_slice, config, htf_df=htf_arg, symbol=symbol)
        except Exception:
            continue
        if not ctx:
            continue

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

        entry_price = float(df.iloc[i]["close"])
        exit_price  = float(df.iloc[i + bars_held]["close"])
        pnl_pct     = round(r_multiple * 1.0, 4)

        signals.append({
            "timestamp":       bar_time.isoformat(),
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

    log.info(f"  Signals found:        {len(signals)}")
    log.info(f"  Skipped (regime):     {skipped_regime}")
    log.info(f"  Skipped (no signal):  {skipped_no_signal}")
    return signals


def _stratified_sample(signals: list[dict], max_samp: int) -> list[dict]:
    """Proportional sample from each outcome class, evenly spaced for temporal diversity."""
    if len(signals) <= max_samp:
        return signals

    by_outcome: dict[str, list] = defaultdict(list)
    for s in signals:
        by_outcome[s["outcome"]].append(s)

    per_class = max(1, max_samp // max(len(by_outcome), 1))
    selected: list[dict] = []
    for _, items in by_outcome.items():
        step = max(1, len(items) // per_class)
        selected.extend(items[::step][:per_class])

    remaining = max_samp - len(selected)
    if remaining > 0:
        selected_set = set(id(s) for s in selected)
        pool = [s for s in signals if id(s) not in selected_set]
        extra = random.sample(pool, min(remaining, len(pool)))
        selected.extend(extra)

    random.shuffle(selected)
    return selected[:max_samp]


def process_symbol(symbol: str, path: str, max_samp: int) -> list[dict]:
    log.info("")
    log.info("=" * 65)
    log.info(f"{symbol}: {path}")
    log.info("=" * 65)

    mid = _load_ticks(path)
    log.info(f"  {len(mid):,} ticks  {mid.index[0]} -> {mid.index[-1]}")

    df     = _resample(mid, _RESAMPLE_RULE[config.TIMEFRAME])
    htf_tf = getattr(config, "HTF_TIMEFRAME", "H4")
    htf_df = _resample(mid, _RESAMPLE_RULE[htf_tf])
    log.info(f"  {len(df)} {config.TIMEFRAME} bars, {len(htf_df)} {htf_tf} bars")

    if len(df) < _WARMUP + 60:
        log.warning(f"  Need at least {_WARMUP + 60} bars, got {len(df)} — skipping {symbol}")
        return []

    signals = _scan_symbol(symbol, df, htf_df)
    if not signals:
        return []

    win_rate = sum(1 for s in signals if s["outcome"] in ("WIN", "WIN_STRONG")) / len(signals)
    log.info(f"  Raw win rate: {win_rate:.1%}")

    sampled = _stratified_sample(signals, max_samp)
    if len(sampled) < len(signals):
        log.info(f"  Stratified sample: {len(sampled)} of {len(signals)}")
    return sampled


def run(args):
    directory = args.dir
    symbols   = [args.symbol] if args.symbol else config.SYMBOLS
    max_samp  = args.max_samples

    store = TradeMemoryStore(
        config.MEMORY_DB_PATH,
        config.MEMORY_USEFULNESS_THRESHOLD,
        config.MEMORY_MAX_STORED,
    )
    log.info(f"Existing memory: {store.stats()}")

    processed  = 0
    all_samples: list[dict] = []
    for symbol in symbols:
        path = _find_csv(directory, symbol)
        if not path:
            continue
        processed += 1
        all_samples.extend(process_symbol(symbol, path, max_samp))

    if not processed:
        log.warning(f"No <SYMBOL>_TICK_*.csv files found in {directory!r} for {symbols}")
        return

    log.info("")
    log.info("=" * 65)
    log.info(f"Writing {len(all_samples)} samples to {config.MEMORY_DB_PATH}...")
    saved = failed = 0
    for s in all_samples:
        if _write_sample(config.MEMORY_DB_PATH, s):
            saved += 1
        else:
            failed += 1

    store._prune_if_needed()

    log.info("")
    log.info(f"Done: {processed} symbols processed, {saved} samples saved, {failed} failed")
    log.info(f"Final memory: {store.stats()}")
    log.info("")
    log.info("Next steps:")
    log.info("  1. python bot.py --demo    (ML gate activates on first tick)")
    log.info("  2. python bot.py           (live trading with pre-trained ML)")
    log.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed bot memory from MT5 tick-export CSVs (no MT5 connection needed)"
    )
    parser.add_argument(
        "--dir", type=str, default=".",
        help="Directory containing <SYMBOL>_TICK_*.csv files (default: cwd)",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Process a single symbol only (default: every symbol in config.SYMBOLS with a matching CSV)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=150, dest="max_samples",
        help="Max training samples per symbol (default: 150)",
    )
    run(parser.parse_args())
