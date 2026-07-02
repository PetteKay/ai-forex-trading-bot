# Forex data loader for the NVIDIA signal discovery workflow.
# Replaces the S&P 500 yfinance loader with major/minor forex pairs.
# yfinance supports forex via the "XXXYYY=X" ticker convention.

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "forex"
DATA_FIELDS = ["Open", "Close", "High", "Low", "Volume"]

FOREX_PAIRS = [
    # Majors
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
    # Minors / crosses
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
    "CADJPY=X", "CHFJPY=X", "EURAUD=X", "EURCHF=X",
    "GBPAUD=X", "GBPCAD=X", "AUDCAD=X", "AUDNZD=X",
    # Exotics (USD pairs)
    "USDZAR=X", "USDNOK=X", "USDSEK=X", "USDMXN=X",
    "USDSGD=X", "USDHKD=X", "USDTRY=X",
]


def get_forex_pairs() -> list[str]:
    logger.info(f"Using {len(FOREX_PAIRS)} forex pairs")
    return FOREX_PAIRS


def download_forex_data(
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    output_dir: str | Path | None = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Download forex OHLCV data via yfinance and save as CSV files
    in the same format expected by the signal discovery workflow.

    Args:
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.
        output_dir: Directory to save CSV files. Defaults to data/forex/.
        interval: yfinance interval string (1d, 1h, etc.).
    """
    output_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    pairs = get_forex_pairs()
    logger.info(f"Downloading {interval} data from {start} to {end} for {len(pairs)} pairs...")

    raw = yf.download(pairs, start=start, end=end, interval=interval, group_by="column", auto_adjust=True)

    results = {}
    for field in DATA_FIELDS:
        if field not in raw.columns.get_level_values(0):
            logger.warning(f"Field '{field}' not found, skipping")
            continue

        df = raw[field].copy()
        df.index.name = "Date"
        # Rename columns: strip the "=X" suffix for readability (EURUSD=X -> EURUSD)
        df.columns = [c.replace("=X", "") for c in df.columns]
        df = df.dropna(axis=1, how="all")

        csv_path = output_path / f"{field}.csv"
        df.to_csv(csv_path)
        results[field] = df
        logger.info(f"Saved {field}.csv  shape={df.shape}")

    logger.info(f"Forex data saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Download forex OHLCV data for signal discovery.")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Output directory. Default: data/forex/")
    parser.add_argument("--interval", default="1d", help="yfinance interval (1d, 1h, 30m). Default: 1d")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    download_forex_data(start=args.start, end=args.end, output_dir=args.output, interval=args.interval)


if __name__ == "__main__":
    main()
