"""
Equity Curve Visualiser.

Generates a matplotlib chart of the equity curve + trade P&L distribution.

Usage:
  python observability/equity_chart.py
  python observability/equity_chart.py --db observability/trade_journal.db
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def plot(db_path: str = "observability/trade_journal.db") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        return

    from observability.journal import Journal
    j = Journal(db_path)

    equity_data = j.get_equity_curve()
    trades      = j.get_recent_trades(500)

    if not equity_data and not trades:
        print("No data yet — run the bot first.")
        return

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("AI Trading Bot — Performance Dashboard", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # ── Equity curve ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    if equity_data:
        times  = [e["timestamp"][:16] for e in equity_data]
        equities = [e["equity"] for e in equity_data]
        ax1.plot(times, equities, color="#2196F3", linewidth=1.5)
        ax1.fill_between(range(len(equities)), equities, min(equities),
                         alpha=0.1, color="#2196F3")
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Account Equity")
        step = max(1, len(times) // 8)
        ax1.set_xticks(range(0, len(times), step))
        ax1.set_xticklabels([times[i] for i in range(0, len(times), step)],
                            rotation=30, fontsize=8)
        ax1.grid(True, alpha=0.3)

    # ── P&L distribution ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    closed = [t for t in trades if t.get("pnl_pct") is not None]
    if closed:
        pnls = [t["pnl_pct"] for t in closed]
        colors = ["#4CAF50" if p > 0 else "#f44336" for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.8)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_title(f"P&L per Trade (last {len(pnls)})")
        ax2.set_ylabel("P&L %")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3, axis="y")

    # ── Win/loss pie ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if closed:
        wins   = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
        losses = len(closed) - wins
        ax3.pie(
            [wins, losses],
            labels=[f"Wins ({wins})", f"Losses ({losses})"],
            colors=["#4CAF50", "#f44336"],
            autopct="%1.1f%%",
            startangle=90,
        )
        win_rate = wins / len(closed) * 100 if closed else 0
        ax3.set_title(f"Win Rate: {win_rate:.1f}%")

    plt.tight_layout()

    output = "observability/equity_curve.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {output}")
    plt.show()
    j.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="observability/trade_journal.db")
    args = parser.parse_args()
    plot(args.db)
