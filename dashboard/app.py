"""
Web Dashboard — live view of bot activity, equity, and trade history.

Run:  py dashboard/app.py
Open: http://127.0.0.1:5000

Reads from observability/trade_journal.db (WAL mode — safe while bot runs).
"""

import sqlite3
from contextlib import contextmanager
from sqlite_conn import connect as _sqlite_connect
import os

from flask import Flask, jsonify, render_template

# Resolve paths relative to the trading-bot root (one level above this file)
_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_ROOT, "observability", "trade_journal.db")

app = Flask(__name__, template_folder="templates")


@contextmanager
def _db():
    with _sqlite_connect(_DB_PATH) as con:     # closed when the block ends
        con.row_factory = sqlite3.Row
        yield con


def _safe_query(sql: str, params: tuple = ()) -> list[dict]:
    if not os.path.exists(_DB_PATH):
        return []
    try:
        with _db() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _safe_scalar(sql: str, params: tuple = (), default=None):
    if not os.path.exists(_DB_PATH):
        return default
    try:
        with _db() as con:
            row = con.execute(sql, params).fetchone()
        return row[0] if row else default
    except Exception:
        return default


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    total   = _safe_scalar("SELECT COUNT(*) FROM trades WHERE status='CLOSED'", default=0)
    wins    = _safe_scalar("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl > 0", default=0)
    losses  = _safe_scalar("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl <= 0", default=0)
    avg_pnl = _safe_scalar("SELECT AVG(pnl) FROM trades WHERE status='CLOSED'", default=0.0)
    total_pnl = _safe_scalar("SELECT SUM(pnl) FROM trades WHERE status='CLOSED'", default=0.0)
    best    = _safe_scalar("SELECT MAX(pnl_pct) FROM trades WHERE status='CLOSED'", default=0.0)
    worst   = _safe_scalar("SELECT MIN(pnl_pct) FROM trades WHERE status='CLOSED'", default=0.0)
    open_ct = _safe_scalar("SELECT COUNT(*) FROM trades WHERE status='OPEN'", default=0)
    last_eq = _safe_scalar("SELECT equity FROM equity_log ORDER BY timestamp DESC LIMIT 1", default=0.0)
    last_tick = _safe_scalar("SELECT timestamp FROM ticks ORDER BY timestamp DESC LIMIT 1", default="—")
    win_rate = round(wins / total * 100, 1) if total else 0.0

    return jsonify({
        "total_trades":  total,
        "wins":          wins,
        "losses":        losses,
        "win_rate_pct":  win_rate,
        "total_pnl":     round(float(total_pnl or 0), 2),
        "avg_pnl":       round(float(avg_pnl or 0), 2),
        "best_pct":      round(float(best or 0), 2),
        "worst_pct":     round(float(worst or 0), 2),
        "open_positions": open_ct,
        "equity":        round(float(last_eq or 0), 2),
        "last_tick":     last_tick,
        "db_exists":     os.path.exists(_DB_PATH),
    })


@app.route("/api/trades")
def api_trades():
    rows = _safe_query(
        "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 50"
    )
    return jsonify(rows)


@app.route("/api/equity")
def api_equity():
    rows = _safe_query(
        "SELECT timestamp, equity, daily_pnl, open_positions FROM equity_log ORDER BY timestamp DESC LIMIT 200"
    )
    rows.reverse()
    return jsonify(rows)


@app.route("/api/ticks")
def api_ticks():
    rows = _safe_query(
        "SELECT timestamp, symbol, regime, tech_bias, pm_action, pm_confidence, session "
        "FROM ticks ORDER BY timestamp DESC LIMIT 100"
    )
    return jsonify(rows)


@app.route("/api/open")
def api_open():
    rows = _safe_query("SELECT * FROM trades WHERE status='OPEN' ORDER BY entry_time DESC")
    return jsonify(rows)


if __name__ == "__main__":
    print(f"Dashboard starting — DB: {_DB_PATH}")
    print(f"DB exists: {os.path.exists(_DB_PATH)}")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, host="127.0.0.1", port=5000)
