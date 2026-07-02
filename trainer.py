"""
Trainer — Pre-load the bot's memory from historical trade data.

Instead of starting with zero knowledge, you can feed the bot a CSV of
past trades and it will extract lessons from each one, building a rich
memory before it ever goes live. This is how you "train" it.

CSV format expected (export from MT5 terminal → History → Save as CSV):
  symbol, direction, entry_price, exit_price, pnl_pct, hold_hours, entry_rsi, ...

OR use the simplified format below — just the essentials.

Usage:
  python trainer.py --file data/my_trades.csv
  python trainer.py --demo          (generates sample trades to show how it works)
"""

import argparse
import csv
import logging
import sys
import os

import config
from agents import base as llm
from agents.reflection import reflect
from memory.store import TradeMemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Trainer")


# ------------------------------------------------------------------ #
# Sample trades (for --demo mode)                                      #
# These show the bot a variety of outcomes to learn from              #
# ------------------------------------------------------------------ #
SAMPLE_TRADES = [
    {
        "symbol": "EURUSD", "direction": "BUY", "entry_price": 1.08520, "exit_price": 1.08920,
        "pnl_pct": 1.8, "hold_hours": 6.5, "outcome": "WIN",
        "context": {"market_structure": "UPTREND", "rsi": 52, "rsi_state": "NEUTRAL",
                    "ema_cross_up": True, "ema_cross_down": False, "fast_above_slow": True,
                    "price_above_trend": True, "macd_positive": True, "bb_pct": 0.65,
                    "bb_squeeze": False, "atr_percentile": 55, "sessions": ["LONDON"]},
        "tech": {"bias": "BUY", "bias_strength": 7, "summary": "Clear uptrend with EMA cross confirmation."},
        "pm": {"action": "BUY", "confidence": 78, "reasoning": "Strong uptrend with multiple confluences."},
    },
    {
        "symbol": "EURUSD", "direction": "BUY", "entry_price": 1.09100, "exit_price": 1.08800,
        "pnl_pct": -1.5, "hold_hours": 3.0, "outcome": "STOP_LOSS",
        "context": {"market_structure": "SIDEWAYS", "rsi": 68, "rsi_state": "NEUTRAL",
                    "ema_cross_up": True, "ema_cross_down": False, "fast_above_slow": True,
                    "price_above_trend": False, "macd_positive": True, "bb_pct": 0.88,
                    "bb_squeeze": False, "atr_percentile": 80, "sessions": ["NEW_YORK"]},
        "tech": {"bias": "BUY", "bias_strength": 5, "summary": "EMA cross but near upper BB in sideways market."},
        "pm": {"action": "BUY", "confidence": 65, "reasoning": "EMA cross confirmed but market structure weak."},
    },
    {
        "symbol": "EURUSD", "direction": "SELL", "entry_price": 1.09300, "exit_price": 1.08900,
        "pnl_pct": 2.2, "hold_hours": 12.0, "outcome": "WIN_STRONG",
        "context": {"market_structure": "DOWNTREND", "rsi": 72, "rsi_state": "OVERBOUGHT",
                    "ema_cross_up": False, "ema_cross_down": True, "fast_above_slow": False,
                    "price_above_trend": False, "macd_positive": False, "bb_pct": 0.92,
                    "bb_squeeze": False, "atr_percentile": 60, "sessions": ["LONDON"]},
        "tech": {"bias": "SELL", "bias_strength": 8, "summary": "Overbought RSI at upper BB in downtrend."},
        "pm": {"action": "SELL", "confidence": 82, "reasoning": "Triple confluence: downtrend + overbought + upper BB rejection."},
    },
    {
        "symbol": "EURUSD", "direction": "BUY", "entry_price": 1.07800, "exit_price": 1.07850,
        "pnl_pct": 0.1, "hold_hours": 1.5, "outcome": "BREAKEVEN",
        "context": {"market_structure": "SIDEWAYS", "rsi": 50, "rsi_state": "NEUTRAL",
                    "ema_cross_up": True, "ema_cross_down": False, "fast_above_slow": True,
                    "price_above_trend": True, "macd_positive": False, "bb_pct": 0.50,
                    "bb_squeeze": True, "atr_percentile": 15, "sessions": ["ASIAN"]},
        "tech": {"bias": "BUY", "bias_strength": 3, "summary": "EMA cross in low-volatility squeeze. Risk of whipsaw."},
        "pm": {"action": "BUY", "confidence": 63, "reasoning": "Marginal setup during squeeze — low conviction."},
    },
    {
        "symbol": "EURUSD", "direction": "SELL", "entry_price": 1.08200, "exit_price": 1.07600,
        "pnl_pct": 3.1, "hold_hours": 18.0, "outcome": "WIN_STRONG",
        "context": {"market_structure": "DOWNTREND", "rsi": 45, "rsi_state": "NEUTRAL",
                    "ema_cross_up": False, "ema_cross_down": True, "fast_above_slow": False,
                    "price_above_trend": False, "macd_positive": False, "bb_pct": 0.55,
                    "bb_squeeze": False, "atr_percentile": 70, "sessions": ["LONDON", "NEW_YORK"]},
        "tech": {"bias": "SELL", "bias_strength": 9, "summary": "Strong downtrend with death cross and MACD below zero."},
        "pm": {"action": "SELL", "confidence": 88, "reasoning": "All indicators aligned bearish. London/NY overlap adds momentum."},
    },
]


def _infer_r_multiple(row_or_dict: dict, pnl_pct: float, outcome: str) -> float:
    """
    Infer R-multiple when not explicitly provided.
    If the CSV has an r_multiple column, use it.
    Otherwise derive from outcome: avoids needing SL distance which is rarely exported.
    """
    raw = row_or_dict.get("r_multiple", "")
    if raw not in (None, "", "nan"):
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    # Outcome-based defaults: represent typical R outcomes for each classification
    defaults = {
        "WIN_STRONG": 1.8,
        "WIN":        0.5,
        "BREAKEVEN":  0.0,
        "LOSS":      -0.5,
        "STOP_LOSS": -1.0,
        "STOP_LOSS_FULL": -1.0,
    }
    return defaults.get(outcome, 0.0)


def run_from_csv(filepath: str, memory: TradeMemoryStore):
    log.info(f"Loading trades from: {filepath}")
    if not os.path.exists(filepath):
        log.error(f"File not found: {filepath}")
        return

    saved = 0
    skipped = 0

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            log.info(f"Processing trade {i+1}: {row.get('direction','?')} {row.get('symbol','?')}")
            # Build minimal context from CSV columns
            ctx = {
                "market_structure": row.get("market_structure", "SIDEWAYS"),
                "rsi": float(row.get("rsi", 50)),
                "rsi_state": row.get("rsi_state", "NEUTRAL"),
                "ema_cross_up": row.get("ema_cross_up", "false").lower() == "true",
                "ema_cross_down": row.get("ema_cross_down", "false").lower() == "true",
                "fast_above_slow": row.get("fast_above_slow", "true").lower() == "true",
                "price_above_trend": row.get("price_above_trend", "true").lower() == "true",
                "macd_positive": row.get("macd_positive", "false").lower() == "true",
                "bb_pct": float(row.get("bb_pct", 0.5)),
                "bb_squeeze": row.get("bb_squeeze", "false").lower() == "true",
                "atr_percentile": float(row.get("atr_percentile", 50)),
                "sessions": row.get("sessions", "LONDON").split("|"),
            }
            tech = {"bias": row.get("direction", "NEUTRAL"), "bias_strength": 5, "summary": "From CSV"}
            pm   = {"action": row.get("direction", "HOLD"), "confidence": int(row.get("confidence", 70)),
                    "reasoning": row.get("notes", "Historical trade")}

            outcome    = row.get("outcome", "WIN")
            pnl_pct    = float(row.get("pnl_pct", 0))
            r_multiple = _infer_r_multiple(row, pnl_pct, outcome)

            lesson_record = reflect(
                symbol            = row.get("symbol", config.SYMBOL),
                direction         = row.get("direction", "BUY"),
                entry_price       = float(row.get("entry_price", 0)),
                exit_price        = float(row.get("exit_price", 0)),
                pnl_pct           = pnl_pct,
                r_multiple        = r_multiple,
                hold_hours        = float(row.get("hold_hours", 4)),
                outcome           = outcome,
                entry_context     = ctx,
                technical_analysis= tech,
                pm_decision       = pm,
                cfg               = config,
            )
            if lesson_record and memory.save(lesson_record):
                saved += 1
                log.info(f"  ✓ Saved: {lesson_record['lesson'][:80]}")
            else:
                skipped += 1

    log.info(f"\nTraining complete: {saved} lessons saved, {skipped} skipped.")
    log.info(f"Memory stats: {memory.stats()}")


def run_demo(memory: TradeMemoryStore):
    log.info("Running demo training with 5 sample trades...")
    saved = 0

    for i, trade in enumerate(SAMPLE_TRADES):
        log.info(f"\nProcessing sample trade {i+1}/{len(SAMPLE_TRADES)}: {trade['direction']} → {trade['outcome']}")
        r_multiple = _infer_r_multiple(trade, trade["pnl_pct"], trade["outcome"])
        lesson_record = reflect(
            symbol             = trade["symbol"],
            direction          = trade["direction"],
            entry_price        = trade["entry_price"],
            exit_price         = trade["exit_price"],
            pnl_pct            = trade["pnl_pct"],
            r_multiple         = r_multiple,
            hold_hours         = trade["hold_hours"],
            outcome            = trade["outcome"],
            entry_context      = trade["context"],
            technical_analysis = trade["tech"],
            pm_decision        = trade["pm"],
            cfg                = config,
        )
        if lesson_record and memory.save(lesson_record):
            saved += 1
            log.info(f"  ✓ Lesson: {lesson_record['lesson'][:100]}")
        else:
            log.info(f"  — Nothing worth saving from this trade.")

    log.info(f"\nDemo training done: {saved}/{len(SAMPLE_TRADES)} lessons saved.")
    log.info(f"Memory stats: {memory.stats()}")
    log.info("\nThe bot now has initial knowledge. Run 'python bot.py --demo' to see it in action.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train bot memory from historical trades")
    parser.add_argument("--file", help="Path to CSV of historical trades")
    parser.add_argument("--demo", action="store_true", help="Run with built-in sample trades")
    args = parser.parse_args()

    llm.init(config)
    memory = TradeMemoryStore(config.MEMORY_DB_PATH, config.MEMORY_USEFULNESS_THRESHOLD, config.MEMORY_MAX_STORED)

    if args.demo:
        run_demo(memory)
    elif args.file:
        run_from_csv(args.file, memory)
    else:
        log.error("Specify --demo or --file path/to/trades.csv")
        parser.print_help()
