"""
Export broker tick CSVs as OHLCV panel files for the signal-discovery workflow.

The NVIDIA signal-discovery evaluator expects five wide CSVs — Open.csv,
Close.csv, High.csv, Low.csv, Volume.csv — where rows are timestamps and
columns are instruments (the same layout its S&P 500 downloader produces).
Rank IC is computed cross-sectionally per row, and needs >= 10 instruments
per timestamp; the 20+ pairs in the local tick exports clear that easily.

Using the broker's own tick data (instead of yfinance daily bars) keeps the
discovered signals on the identical price distribution the bot trades.

USAGE
  python export_discovery_panel.py                # H1 bars -> signal-discovery data dir
  python export_discovery_panel.py --rule 4h      # different bar size
  python export_discovery_panel.py --out my/dir   # custom output directory
"""

import argparse
import glob
import logging
import os
import sys

import pandas as pd

from csv_historical_trainer import _load_ticks, _resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("PanelExport")

DEFAULT_OUT = os.path.join(
    "signal-discovery", "src", "signal_discovery_workflow", "data", "forex"
)

FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def build_panels(directory: str = ".", rule: str = "1h") -> dict[str, pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(directory, "*_TICK_*.csv")))
    if not paths:
        raise SystemExit(f"No *_TICK_*.csv files found in {directory!r}")

    per_symbol: dict[str, pd.DataFrame] = {}
    for path in paths:
        symbol = os.path.basename(path).split("_TICK_")[0]
        mid = _load_ticks(path)
        bars = _resample(mid, rule)
        per_symbol[symbol] = bars
        log.info(f"{symbol}: {len(mid):,} ticks -> {len(bars)} {rule} bars")

    panels: dict[str, pd.DataFrame] = {}
    for out_field, bar_col in FIELDS.items():
        panel = pd.DataFrame({sym: bars[bar_col] for sym, bars in per_symbol.items()})
        panel.index.name = "Date"
        panels[out_field] = panel
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir",  default=".", help="Directory with <SYMBOL>_TICK_*.csv files")
    parser.add_argument("--rule", default="1h", help="Bar size (pandas rule: 1h, 4h, 1D)")
    parser.add_argument("--out",  default=DEFAULT_OUT, help="Output directory for panel CSVs")
    args = parser.parse_args()

    panels = build_panels(args.dir, args.rule)
    os.makedirs(args.out, exist_ok=True)
    for field, panel in panels.items():
        path = os.path.join(args.out, f"{field}.csv")
        panel.to_csv(path)
        log.info(f"Saved {path}  shape={panel.shape}")

    n_rows, n_cols = next(iter(panels.values())).shape
    if n_cols < 10:
        log.warning(
            f"Only {n_cols} instruments — the evaluator skips timestamps with "
            f"< 10 instruments, so Rank IC may not compute."
        )
    log.info(f"Done: {len(panels)} panels, {n_rows} rows x {n_cols} symbols -> {args.out}")


if __name__ == "__main__":
    main()
