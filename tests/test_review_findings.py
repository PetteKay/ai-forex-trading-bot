"""
Codex review findings on PR #4 / #5, reproduced and fixed.

  F1  trade_booking: a trade closed with no entry snapshot was booked as BREAKEVEN
      whatever its P&L (risk unknown -> R = 0).
  F2  bot.py: a bot started in demo mode refused close requests sent over MCP in
      live mode — and, holding the account lock, kept the MCP server from closing.
  F3  trade_booking: if the exit time could not be converted to UTC, a loss on a
      trade opened before today was not counted toward today's loss cap.

Run: py -m pytest tests/test_review_findings.py -v
"""
import importlib
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

importlib.import_module("numpy")   # load once, before any module juggling
import bot_control as bc  # noqa: E402


class _Gov:
    def __init__(self):
        self.closed = []

    def record_trade_closed(self, pnl):
        self.closed.append(pnl)


class _Booking(unittest.TestCase):
    """book_closed_trades with the real journal and memory store in a temp folder."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = os.getcwd()
        os.chdir(self.tmp)
        import config
        from observability.journal import Journal
        from memory.store import TradeMemoryStore
        self.cfg = config
        self.j = Journal(os.path.join(self.tmp, "journal.db"))
        self.m = TradeMemoryStore(os.path.join(self.tmp, "memory.db"),
                                  config.MEMORY_USEFULNESS_THRESHOLD, config.MEMORY_MAX_STORED)
        self.gov = _Gov()

    def tearDown(self):
        os.chdir(self.old)

    def open_trade(self, ticket, entry_days_ago=0, sl=1.0950, entry=1.1000, volume=0.10):
        self.j.log_trade_open(ticket=ticket, symbol="EURUSD", direction="BUY", entry_price=entry,
                              sl=sl, tp=1.1100, volume=volume, confidence=60, regime="RANGING",
                              fingerprint="fp", size_multiplier=1.0)
        when = (datetime.now(timezone.utc) - timedelta(days=entry_days_ago)).isoformat()
        from sqlite_conn import connect
        with connect(os.path.join(self.tmp, "journal.db")) as con:
            con.execute("UPDATE trades SET entry_time=? WHERE ticket=?", (when, ticket))

    def mt5b(self, net, exit_utc_known=True, fresh_symbols=("EURUSD",), risk=None):
        now = datetime.now(timezone.utc)

        def t2u(ts, sym, now_utc=None):
            if not exit_utc_known or sym not in fresh_symbols:
                return None
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        return types.SimpleNamespace(
            get_all_open_positions=lambda magic: [],
            get_account_equity=lambda: 10_000.0,
            get_closed_trade=lambda t, max_wait_secs=0: {
                "ticket": t, "exit_price": 1.0900, "net_pnl": net,
                "exit_time": now.replace(tzinfo=None)},
            terminal_time_to_utc=t2u,
            get_risk_usd=(lambda s, e, sl, v: risk) if risk is not None else
                         (lambda s, e, sl, v: (_ for _ in ()).throw(RuntimeError("no MT5"))),
        )

    def book(self, mt5b, symbols=("EURUSD",)):
        from trade_booking import book_closed_trades
        old = getattr(self.cfg, "SYMBOLS", None)
        self.cfg.SYMBOLS = list(symbols)
        try:
            return book_closed_trades(journal=self.j, memory=self.m, governor=self.gov,
                                      meta_gate=None, mt5b=mt5b, magic=1, cfg=self.cfg)
        finally:
            self.cfg.SYMBOLS = old


class F1_NoSnapshot(_Booking):
    def test_risk_recovered_from_the_journal_stop(self):
        self.open_trade(1)
        b = self.book(self.mt5b(-75.0, risk=50.0))[0]
        self.assertEqual(b["r_multiple"], -1.5)
        self.assertEqual(b["outcome"], "STOP_LOSS")
        row = next(r for r in self.j.get_recent_trades(5) if r["ticket"] == 1)
        self.assertEqual(row["outcome"], "STOP_LOSS")

    def test_sign_fallback_when_risk_cannot_be_recovered(self):
        self.open_trade(2, sl=0.0)
        b = self.book(self.mt5b(-75.0))[0]
        self.assertEqual(b["outcome"], "LOSS", "a loss was booked as " + b["outcome"])
        self.assertIsNone(b["r_multiple"])
        self.open_trade(3, sl=0.0)
        self.assertEqual(self.book(self.mt5b(40.0))[0]["outcome"], "WIN")


class F3_UnknownExitTime(_Booking):
    def test_loss_on_older_trade_counts_today_when_exit_time_unknown(self):
        self.open_trade(10, entry_days_ago=2)
        b = self.book(self.mt5b(-80.0, exit_utc_known=False))[0]
        self.assertTrue(b["counted_in_today"])
        self.assertEqual(self.gov.closed, [-80.0])

    def test_profit_with_unknown_exit_time_is_not_counted(self):
        # counting it could offset today's losses and loosen the cap
        self.open_trade(11, entry_days_ago=2)
        self.book(self.mt5b(+80.0, exit_utc_known=False))
        self.assertEqual(self.gov.closed, [])

    def test_another_symbols_fresh_tick_gives_the_exit_time(self):
        # EURUSD tick stale, GBPUSD fresh: same terminal clock, so use it
        self.open_trade(12, entry_days_ago=2)
        b = self.book(self.mt5b(+80.0, fresh_symbols=("GBPUSD",)), symbols=("EURUSD", "GBPUSD"))[0]
        self.assertTrue(b["counted_in_today"])
        self.assertEqual(self.gov.closed, [80.0])


class F2_DemoBotClose(unittest.TestCase):
    """The real bot.run() in demo mode carries out a close request from MCP."""

    def test_demo_bot_closes_on_request(self):
        from tests.tick_harness import run_one_tick
        wd = tempfile.mkdtemp()
        holder = {}

        def make():
            b = bc.BotSide("demo")
            assert b.acquire(wait=0)[0]
            holder["b"] = b
            return b
        pos = [{"ticket": 88, "symbol": "EURUSD", "type": "BUY", "volume": 0.1, "price": 1.1,
                "sl": 1.09, "tp": 1.12, "profit": 0.0, "time": 0}]

        def ask(n, config, positions):
            if n == 1:
                bc.request_close(bc.read_status()["instance"], 88)
        try:
            r = run_one_tick(n_symbols=1, latency=0, ticks=2, start_positions=pos,
                             on_tick_end=ask, control=make, workdir=wd, demo=True)
        finally:
            if "b" in holder:
                holder["b"].release()
        self.assertIsNone(r["error"], r["error"])
        self.assertEqual(r["closes"], [88])
        st = bc.read_status(os.path.join(wd, bc.STATUS_PATH))
        self.assertEqual(st["closes"], {"88": "closed"})


if __name__ == "__main__":
    unittest.main()
