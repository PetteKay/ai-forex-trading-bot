"""
Daily Performance Reporter.

Generates a P&L breakdown by symbol and regime from the trade journal.
Triggered at midnight UTC crossing or on bot shutdown.

Usage:
  from observability.reporter import generate_daily_report
  report = generate_daily_report("observability/trade_journal.db", days=1)
"""

import sqlite3
import os
import logging
from datetime import datetime, timezone, timedelta

log = logging.getLogger("Reporter")


def generate_daily_report(
    db_path: str,
    days: int = 1,
    send_telegram=None,
) -> str:
    """
    Generate a performance breakdown for the last N days.

    Args:
        db_path:        Path to trade_journal.db
        days:           Look back this many days (1 = today, 7 = last week)
        send_telegram:  Optional callable(text) — pushes report to Telegram

    Returns:
        Report as a multi-line string (also logged at INFO level).
    """
    if not os.path.exists(db_path):
        return "No journal database found."

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row

        rows = con.execute("""
            SELECT symbol, regime, pnl, pnl_pct, outcome
            FROM trades
            WHERE status='CLOSED' AND exit_time >= ?
        """, (cutoff,)).fetchall()
        con.close()

        if not rows:
            msg = f"No closed trades in the last {days} day(s)."
            log.info(msg)
            return msg

        total     = len(rows)
        wins      = sum(1 for r in rows if (r["pnl"] or 0) > 0)
        total_pnl = sum(r["pnl"] or 0.0 for r in rows)
        win_rate  = wins / total * 100 if total else 0.0

        label = "Today" if days == 1 else f"Last {days} days"
        lines = [
            "=" * 52,
            f"  PERFORMANCE REPORT — {label}",
            f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 52,
            f"  Trades : {total}   Wins : {wins}   Win Rate : {win_rate:.1f}%",
            f"  Total P&L : {total_pnl:+.2f}",
            "",
        ]

        # ── By symbol ────────────────────────────────────────────────────────
        sym_stats: dict[str, dict] = {}
        for r in rows:
            sym = r["symbol"] or "?"
            if sym not in sym_stats:
                sym_stats[sym] = {"n": 0, "w": 0, "pnl": 0.0}
            sym_stats[sym]["n"]   += 1
            sym_stats[sym]["pnl"] += r["pnl"] or 0.0
            if (r["pnl"] or 0) > 0:
                sym_stats[sym]["w"] += 1

        lines.append("  BY SYMBOL:")
        for sym, s in sorted(sym_stats.items(), key=lambda x: -x[1]["pnl"]):
            wr = s["w"] / s["n"] * 100 if s["n"] else 0
            lines.append(
                f"    {sym:<10}  {s['n']:>3} trades  "
                f"WR {wr:>5.1f}%  P&L {s['pnl']:>+8.2f}"
            )

        lines.append("")

        # ── By regime ────────────────────────────────────────────────────────
        reg_stats: dict[str, dict] = {}
        for r in rows:
            reg = (r["regime"] or "UNKNOWN").split("_")[0][:10]
            if reg not in reg_stats:
                reg_stats[reg] = {"n": 0, "w": 0, "pnl": 0.0}
            reg_stats[reg]["n"]   += 1
            reg_stats[reg]["pnl"] += r["pnl"] or 0.0
            if (r["pnl"] or 0) > 0:
                reg_stats[reg]["w"] += 1

        lines.append("  BY REGIME:")
        for reg, s in sorted(reg_stats.items(), key=lambda x: -x[1]["pnl"]):
            wr = s["w"] / s["n"] * 100 if s["n"] else 0
            lines.append(
                f"    {reg:<12}  {s['n']:>3} trades  "
                f"WR {wr:>5.1f}%  P&L {s['pnl']:>+8.2f}"
            )

        # ── Outcome distribution ──────────────────────────────────────────────
        lines.append("")
        lines.append("  OUTCOME BREAKDOWN:")
        outcome_counts: dict[str, int] = {}
        for r in rows:
            oc = r["outcome"] or "?"
            outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
        for oc, n in sorted(outcome_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {oc:<16}  {n:>3}")

        lines.append("=" * 52)

        report = "\n".join(lines)
        log.info("\n" + report)

        if send_telegram and callable(send_telegram):
            try:
                send_telegram(f"```\n{report}\n```")
            except Exception as _e:
                log.debug(f"Reporter Telegram send failed: {_e}")

        return report

    except Exception as e:
        log.warning(f"Daily report generation failed: {e}")
        return f"Report failed: {e}"
