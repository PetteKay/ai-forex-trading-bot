"""
bot_control.py, bot.py and the MCP control tools against specs/BotControl.tla.

Each class names the spec invariants it checks on the real code:
  Overrides            INV-BC-6  limits only tighten (incl. a later config.py edit)
  Locks                INV-BC-1/2 one trader per account
  ControlProtocol      INV-BC-4/5 stop names its instance; confirmed pause
  ModelBased           INV-BC-1/3/4/5 on random action sequences that mirror the
                       spec's actions, executed on the real locks and control files
  BotLoop              the real bot.run(): pause (before and mid-tick), stop, close,
                       retrain, trades closed while no bot ran booked once, overrides
  McpProcesses         real processes through the MCP tools: start / status / pause /
                       close / retrain / stop / force / crash; bot.py refuses a 2nd copy

Run: py -m pytest tests/test_bot_control.py -v
"""
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = os.path.join(ROOT, "tests", "fake_mt5")
sys.path.insert(0, ROOT)

import importlib  # noqa: E402
for _m in ("numpy", "pandas"):   # load once, before any module juggling
    importlib.import_module(_m)
import bot_control as bc  # noqa: E402


class _InTmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.old)


# ── INV-BC-6 ─────────────────────────────────────────────────────────────────
class Overrides(_InTmp):
    def cfg(self, **kw):
        base = dict(RISK_PER_TRADE_PCT=0.5, MAX_LOT_SIZE=1.0, MAX_OPEN_POSITIONS=3,
                    MAX_ORDERS_PER_DAY=8, DAILY_LOSS_CAP_PCT=2.0, MAX_DRAWDOWN_PCT=10.0,
                    META_LABEL_MIN_PROB=0.52, MIN_CONFIDENCE_TO_TRADE=50,
                    SYMBOLS=["EURUSD", "GBPUSD", "USDJPY"])
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_only_tightening_is_accepted(self):
        c = self.cfg()
        ok = bc.set_overrides(c, {"RISK_PER_TRADE_PCT": 0.25, "MAX_OPEN_POSITIONS": 2,
                                  "META_LABEL_MIN_PROB": 0.6, "SYMBOLS": ["USDJPY", "EURUSD"]})
        self.assertTrue(ok["saved"], ok)
        bc.apply_overrides(c)
        self.assertEqual((c.RISK_PER_TRADE_PCT, c.MAX_OPEN_POSITIONS, c.META_LABEL_MIN_PROB),
                         (0.25, 2, 0.6))
        self.assertEqual(c.SYMBOLS, ["EURUSD", "USDJPY"])           # config.py order kept
        for bad in ({"RISK_PER_TRADE_PCT": 1.0}, {"MAX_OPEN_POSITIONS": 4},
                    {"META_LABEL_MIN_PROB": 0.4}, {"MIN_CONFIDENCE_TO_TRADE": 10},
                    {"SYMBOLS": ["XAUUSD"]}, {"SYMBOLS": []}, {"RISK_PER_TRADE_PCT": 0},
                    {"RISK_PER_TRADE_PCT": float("nan")}, {"MAX_OPEN_POSITIONS": 1.5},
                    {"META_LABEL_MIN_PROB": 0.99}, {"MCP_TRADING_MODE": "live"},
                    {"MCP_HTTP_ALLOW_TRADING": True}, {"LLM_API_KEY": "x"}):
            r = bc.set_overrides(c, bad)
            self.assertFalse(r["saved"], bad)

    def test_one_bad_key_saves_nothing(self):
        c = self.cfg()
        r = bc.set_overrides(c, {"MAX_OPEN_POSITIONS": 1, "MCP_TRADING_MODE": "live"})
        self.assertFalse(r["saved"])
        self.assertEqual(bc.read_overrides(), {})

    def test_later_config_edit_still_caps_an_older_override(self):
        """Mutation M3 in the spec: using the override directly would let it exceed
        a value the human lowered later."""
        c = self.cfg()
        bc.set_overrides(c, {"RISK_PER_TRADE_PCT": 0.4})
        c2 = self.cfg(RISK_PER_TRADE_PCT=0.3)              # human edited config.py, restart
        bc.apply_overrides(c2)
        self.assertEqual(c2.RISK_PER_TRADE_PCT, 0.3)
        c3 = self.cfg(SYMBOLS=["EURUSD"])
        bc.set_overrides(c, {"SYMBOLS": ["GBPUSD", "USDJPY"]})
        bc.apply_overrides(c3)
        self.assertEqual(c3.SYMBOLS, ["EURUSD"])           # never empty, never wider

    def test_clearing_restores_config_py_in_a_running_process(self):
        c = self.cfg()
        bc.set_overrides(c, {"MAX_OPEN_POSITIONS": 1})
        bc.apply_overrides(c)
        self.assertEqual(c.MAX_OPEN_POSITIONS, 1)
        bc.clear_overrides()
        bc.apply_overrides(c)
        self.assertEqual(c.MAX_OPEN_POSITIONS, 3)

    def test_value_changed_by_someone_else_is_the_new_config_value(self):
        """Regression: a value set elsewhere that equals what we last applied
        (with no override in effect) must not bring back an old config value."""
        c = self.cfg(META_LABEL_MIN_PROB=0.52)
        bc.apply_overrides(c)
        c.META_LABEL_MIN_PROB = 0.0           # e.g. a test, or an edit + reload
        bc.apply_overrides(c)
        self.assertEqual(c.META_LABEL_MIN_PROB, 0.0)
        c.META_LABEL_MIN_PROB = 0.52
        bc.apply_overrides(c)
        c.META_LABEL_MIN_PROB = 0.0
        bc.apply_overrides(c)
        self.assertEqual(c.META_LABEL_MIN_PROB, 0.0)
        bc.set_overrides(c, {"META_LABEL_MIN_PROB": 0.6})
        bc.apply_overrides(c)
        bc.apply_overrides(c)                 # re-applying keeps config.py's 0.0 as the base
        self.assertEqual(c._FILE_VALUES["META_LABEL_MIN_PROB"], 0.0)
        self.assertEqual(c.META_LABEL_MIN_PROB, 0.6)

    def test_random_sequences_never_loosen(self):
        rng = random.Random(7)
        for _ in range(300):
            human = {"RISK_PER_TRADE_PCT": rng.choice([0.2, 0.5, 1.0]),
                     "META_LABEL_MIN_PROB": rng.choice([0.5, 0.55, 0.6])}
            c = self.cfg(**human)
            bc.clear_overrides()
            for _ in range(6):
                act = rng.random()
                if act < 0.4:
                    bc.set_overrides(c, {"RISK_PER_TRADE_PCT": rng.choice([0.1, 0.3, 0.6, 2.0])})
                elif act < 0.7:
                    bc.set_overrides(c, {"META_LABEL_MIN_PROB": rng.choice([0.45, 0.58, 0.7, 0.9])})
                elif act < 0.85:
                    bc.clear_overrides()
                else:                                         # [HumanEdit] + restart
                    human = {"RISK_PER_TRADE_PCT": rng.choice([0.2, 0.5, 1.0]),
                             "META_LABEL_MIN_PROB": rng.choice([0.5, 0.55, 0.6])}
                    c = self.cfg(**human)
                bc.apply_overrides(c)
                self.assertLessEqual(c.RISK_PER_TRADE_PCT, human["RISK_PER_TRADE_PCT"])
                self.assertGreaterEqual(c.META_LABEL_MIN_PROB, human["META_LABEL_MIN_PROB"])


# ── INV-BC-1/2 ───────────────────────────────────────────────────────────────
class Locks(_InTmp):
    def test_running_bot_excludes_mcp_and_a_second_bot(self):
        a = bc.BotSide("demo")
        self.assertTrue(a.acquire(wait=0)[0])
        try:
            b = bc.BotSide("demo")
            self.assertFalse(b.acquire(wait=0.3)[0])
            with bc.account_lock().hold() as mine:
                self.assertFalse(mine)
        finally:
            a.release()
        with bc.account_lock().hold() as mine:
            self.assertTrue(mine)
            b = bc.BotSide("demo")
            self.assertFalse(b.acquire(wait=0.3)[0], "bot started while MCP held the lock")
        self.assertTrue(b.acquire(wait=0)[0])
        b.release()

    def test_mcp_lock_is_reentrant_and_serialises_threads(self):
        lk = bc.account_lock()
        inside, overlap = [0], []

        def work():
            with lk.hold() as mine:
                self.assertTrue(mine)
                with lk.hold() as again:                # nested (place -> plan -> book)
                    self.assertTrue(again)
                inside[0] += 1
                overlap.append(inside[0])
                time.sleep(0.01)
                inside[0] -= 1
        ts = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(max(overlap), 1)
        self.assertFalse(lk.held_here)


# ── INV-BC-4/5 ───────────────────────────────────────────────────────────────
class ControlProtocol(_InTmp):
    def test_stop_names_one_instance(self):
        old = bc.BotSide("demo")
        old.acquire(wait=0)
        bc.request_stop(old.instance)
        self.assertTrue(old.poll()["stop"])
        old.release()
        new = bc.BotSide("demo")
        new.acquire(wait=0)
        self.assertFalse(new.poll()["stop"], "a stale stop request stopped a new bot")
        new.release()

    def test_pause_is_confirmed_through_the_status_file(self):
        b = bc.BotSide("demo")
        b.acquire(wait=0)
        c = bc.set_paused(True)
        self.assertNotEqual(bc.read_status().get("control_version"), c["version"])
        self.assertTrue(b.poll()["paused"])
        st = bc.read_status()
        self.assertEqual(st["control_version"], c["version"])
        self.assertTrue(st["paused"])
        b.release()

    def test_concurrent_writers_lose_no_update(self):
        b = bc.BotSide("demo")
        b.acquire(wait=0)

        def w(k):
            for i in range(25):
                bc.request_close(b.instance, k * 100 + i)
        ts = [threading.Thread(target=w, args=(k,)) for k in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        c = bc.read_control()
        self.assertEqual(c["version"], 100)
        self.assertEqual(len(b.poll()["close"]), 50)          # file keeps the latest 50
        b.release()


# ── Random walks over the spec's actions, on the real code ───────────────────
class ModelBased(_InTmp):
    """Each step picks an enabled action of BotControl.tla and performs it with the
    real locks / control file; after every step the spec's invariants are checked
    on what the code actually did."""

    def walk(self, seed, steps=150):
        rng = random.Random(seed)
        bots = {}                 # instance -> BotSide (running)
        exited = {}               # instance -> cause
        trades, booked = {}, {}
        mcp_lock = bc.account_lock()
        confirmed_pause = False
        requested = set()         # instances a stop request was sent to
        for _ in range(steps):
            act = rng.choice(["launch", "crash", "stop_req", "bot_poll", "pause", "resume",
                              "bot_order", "mcp_place", "market_close", "bot_book", "mcp_book"])
            if act == "launch" and len(bots) + len(exited) < 4:            # [Launch/Acquire/LoseRace]
                b = bc.BotSide("demo")
                ok, _ = b.acquire(wait=0)
                if ok:
                    bots[b.instance] = b
                else:
                    exited[b.instance] = "lostlock"
            elif act == "crash" and bots:                                    # [Crash]
                inst = rng.choice(list(bots))
                bots.pop(inst)._lock.release()
                exited[inst] = "crash"
            elif act == "stop_req" and bots:                                 # [RequestStop]
                inst = rng.choice(list(bots))
                bc.request_stop(inst)
                requested.add(inst)
            elif act == "bot_poll" and bots:                                 # [BotReadControl / GracefulStop]
                inst = rng.choice(list(bots))
                cmd = bots[inst].poll()
                confirmed_pause = cmd["paused"]
                if cmd["stop"]:
                    self.assertIn(inst, bc.read_control()["stop"])         # INV-BC-4
                    bots.pop(inst).release()
                    exited[inst] = "stop"
            elif act in ("pause", "resume"):                                 # [SetPause]
                bc.set_paused(act == "pause")
                confirmed_pause = False
            elif act == "bot_order" and bots:                                # [BotBeginOrder/Fill]
                inst = rng.choice(list(bots))
                cmd = bots[inst].poll()
                if not (cmd["paused"] or cmd["stop"]):
                    self.assertFalse(confirmed_pause and bc.read_control()["paused"])  # INV-BC-5
                    t = len(trades) + 1
                    trades[t] = "open"
            elif act == "mcp_place":                                         # [McpPlace]
                with mcp_lock.hold() as mine:
                    self.assertEqual(mine, not bots)                         # INV-BC-1/2
                    if mine and not bc.read_control()["paused"]:
                        trades[len(trades) + 1] = "open"
            elif act == "market_close" and trades:                           # [StopOrTargetHit]
                t = rng.choice(list(trades))
                if trades[t] == "open":
                    trades[t] = "closed"
            elif act == "bot_book" and bots:                                 # [BotBook]
                for t, s in trades.items():
                    if s == "closed":
                        trades[t] = "booked"
                        booked[t] = booked.get(t, 0) + 1
            elif act == "mcp_book":                                          # [McpBook]
                with mcp_lock.hold() as mine:
                    if mine:
                        for t, s in trades.items():
                            if s == "closed":
                                trades[t] = "booked"
                                booked[t] = booked.get(t, 0) + 1
            # invariants
            self.assertLessEqual(len(bots), 1)                               # INV-BC-1
            self.assertTrue(all(v == 1 for v in booked.values()), booked)    # INV-BC-3
            for inst, cause in exited.items():                               # INV-BC-4
                if cause == "stop":
                    self.assertIn(inst, requested)
        for b in bots.values():
            b.release()

    def test_random_walks(self):
        import shutil
        for seed in range(40):
            shutil.rmtree("observability", ignore_errors=True)
            self.walk(seed)


# ── The real bot.run() loop ──────────────────────────────────────────────────
from tests.tick_harness import run_one_tick  # noqa: E402


class BotLoop(unittest.TestCase):
    def side(self):
        holder = {}

        def make():
            b = bc.BotSide("live", account="12345")
            assert b.acquire(wait=0)[0]
            holder["b"] = b
            return b
        return make, holder

    def tearDown(self):
        for h in getattr(self, "_holders", []):
            if "b" in h:
                h["b"].release()

    def run_bot(self, **kw):
        make, holder = self.side()
        self._holders = getattr(self, "_holders", []) + [holder]
        r = run_one_tick(n_symbols=3, latency=0, buy_every=1, control=make, **kw)
        self.assertIsNone(r["error"], r["error"])
        return r, holder["b"]

    def test_baseline_places_orders(self):
        r, _ = self.run_bot()
        self.assertTrue(r["orders"])

    def test_pause_before_tick_blocks_new_entries(self):
        wd = tempfile.mkdtemp()
        old = os.getcwd()
        os.chdir(wd)
        bc.set_paused(True)
        os.chdir(old)
        r, b = self.run_bot(workdir=wd)
        self.assertEqual(r["orders"], [])
        self.assertEqual(r["llm_calls"].get("technical", 0), 0, "desk ran while paused")
        st = bc.read_status(os.path.join(wd, bc.STATUS_PATH))
        self.assertTrue(st["paused"])

    def test_pause_arriving_mid_tick_stops_the_order(self):
        wd = tempfile.mkdtemp()

        def pause_now(sym):
            bc.set_paused(True)          # cwd is the bot's work folder during run()
        r, _ = self.run_bot(workdir=wd, before_pm=pause_now)
        self.assertGreater(r["llm_calls"].get("portfolio_manager", 0), 0)
        self.assertEqual(r["orders"], [], "an order went out after the pause")

    def test_stop_request_ends_the_loop_and_a_stale_one_does_not(self):
        wd = tempfile.mkdtemp()

        def stop_after_first(n, config, pos):
            if n == 1:
                bc.request_stop(bc.read_status()["instance"])
        r, _ = self.run_bot(workdir=wd, ticks=5, on_tick_end=stop_after_first)
        self.assertEqual(r["ticks_run"], 1)
        wd2 = tempfile.mkdtemp()

        def stale(n, config, pos):
            if n == 1:
                bc.request_stop("0123456789ab")
        r2, _ = self.run_bot(workdir=wd2, ticks=3, on_tick_end=stale)
        self.assertEqual(r2["ticks_run"], 3)

    def test_close_and_retrain_requests_are_carried_out(self):
        wd = tempfile.mkdtemp()
        pos = [{"ticket": 77, "symbol": "EURUSD", "type": "BUY", "volume": 0.1, "price": 1.1,
                "sl": 1.09, "tp": 1.12, "profit": 0.0, "time": 0}]

        def ask(n, config, positions):
            if n == 1:
                inst = bc.read_status()["instance"]
                bc.request_close(inst, 77)
                bc.request_retrain(inst)
        r, b = self.run_bot(workdir=wd, ticks=2, start_positions=pos, on_tick_end=ask,
                            cfg_extra={"MAX_OPEN_POSITIONS": 1})
        self.assertEqual(r["closes"], [77])
        st = bc.read_status(os.path.join(wd, bc.STATUS_PATH))
        self.assertEqual(st["closes"], {"77": "closed"})
        self.assertEqual(st["retrain_done"], 1)
        self.assertIn("trained", st["retrain_result"])

    def test_trade_closed_while_no_bot_ran_is_booked_once(self):
        wd = tempfile.mkdtemp()
        old = os.getcwd()
        os.chdir(wd)
        try:
            import config
            from observability.journal import Journal
            from memory.store import TradeMemoryStore
            j = Journal(config.JOURNAL_DB_PATH)
            j.log_trade_open(ticket=555, symbol="EURUSD", direction="BUY", entry_price=1.1,
                             sl=1.09, tp=1.12, volume=0.1, confidence=70, regime="RANGING",
                             fingerprint="fp", size_multiplier=1.0)
            m = TradeMemoryStore(config.MEMORY_DB_PATH, config.MEMORY_USEFULNESS_THRESHOLD,
                                 config.MEMORY_MAX_STORED)
            m.save_open_context(555, symbol="EURUSD", direction="BUY",
                                entry_time="2026-01-05T10:00:00+00:00", entry_price=1.1,
                                risk_usd=20.0, context={"rsi": 40.0, "atr": 0.001,
                                                        "regime": {"regime": "RANGING"}},
                                source="mcp", reason="test")
        finally:
            os.chdir(old)
        r, _ = self.run_bot(workdir=wd, ticks=3, closed_history={555: -30.0},
                            cfg_extra={"MAX_OPEN_POSITIONS": 0})
        os.chdir(wd)
        try:
            rows = [x for x in j.get_recent_trades(50) if x["ticket"] == 555]
            self.assertEqual(rows[0]["status"], "CLOSED")
            from sqlite_conn import connect
            with connect(config.MEMORY_DB_PATH) as con:
                n = con.execute("SELECT COUNT(*) FROM ml_samples WHERE ticket=555").fetchone()[0]
            self.assertEqual(n, 1)
            gpath = [os.path.join("observability", f) for f in os.listdir("observability")
                     if f.startswith("governor_state")][0]
            with open(gpath) as fh:
                gov = json.load(fh)
            # close time unknown here, so it counts on the entry day (today): once, not 3x
            self.assertEqual(float(gov["daily_pnl"]), -30.0)
        finally:
            os.chdir(old)

    def test_settings_overrides_reach_the_bot(self):
        wd = tempfile.mkdtemp()
        old = os.getcwd()
        os.chdir(wd)
        try:
            import config
            with mock.patch.object(config, "SYMBOLS", ["EURUSD", "GBPUSD", "USDJPY"]):
                self.assertTrue(bc.set_overrides(config, {"SYMBOLS": ["USDJPY"]})["saved"])
        finally:
            os.chdir(old)
        r, _ = self.run_bot(workdir=wd)
        self.assertEqual(r["llm_calls"].get("technical"), 1, r["llm_calls"])
        self.assertEqual(r["orders"], ["USDJPY"])


# ── Real processes through the MCP tools ─────────────────────────────────────
def _load_fake_mt5():
    spec = importlib.util.spec_from_file_location("MetaTrader5", os.path.join(FAKE, "MetaTrader5.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class McpProcesses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp)
        self.fake = _load_fake_mt5()
        self._had = sys.modules.get("MetaTrader5")
        sys.modules["MetaTrader5"] = self.fake
        for m in ("mt5_bridge", "position_manager", "mcp_server"):
            sys.modules.pop(m, None)
        import config
        self.config = config
        self.patches = [mock.patch.object(config, k, v, create=True) for k, v in dict(
            LIVE_CALENDAR_ENABLED=False, CALENDAR_CSV_PATH=os.path.join(self.tmp, "none.csv"),
            ALLOWED_SESSIONS=[], SPREAD_FILTER_ENABLED=False, SIZING_MODE="risk_pct",
            RISK_PER_TRADE_PCT=0.5, MCP_TRADING_MODE="live", MCP_HTTP_ALLOW_TRADING=False,
            MAX_OPEN_POSITIONS=5, MAX_ORDERS_PER_DAY=10, META_LABEL_MIN_PROB=0.0,
            LOSS_STREAK_ENABLED=False, VOL_ADJUSTED_SIZING=False, MCP_LESSONS="client",
            LLM_API_KEY="", SETUP_BLACKLIST_ENABLED=False, TELEGRAM_ALERTS_ENABLED=False).items()]
        for p in self.patches:
            p.start()
        import mcp_server
        self.srv = mcp_server
        self.script = mock.patch.object(mcp_server, "_BOT_SCRIPT", os.path.join(ROOT, "tests", "fake_bot.py"))
        self.script.start()
        self.env = mock.patch.dict(os.environ, {"PYTHONPATH": os.pathsep.join([FAKE, ROOT])})
        self.env.start()
        sys.modules["data.macro_context"] = type(sys)("data.macro_context")
        sys.modules["data.macro_context"].get_macro_context = lambda s: {}

    def tearDown(self):
        try:
            if bc.bot_state()["running"]:
                self.srv.stop_bot(force=True, wait_seconds=3)
        finally:
            for p in list(self.srv._CHILDREN.values()):
                if p.poll() is None:
                    p.kill()
                    p.wait(5)
            self.env.stop()
            self.script.stop()
            sys.modules.pop("data.macro_context", None)
            for p in reversed(self.patches):
                p.stop()
            if self._had is not None:
                sys.modules["MetaTrader5"] = self._had
            else:
                sys.modules.pop("MetaTrader5", None)
            for m in ("mt5_bridge", "position_manager", "mcp_server"):
                sys.modules.pop(m, None)
            os.chdir(self.old_cwd)

    def test_start_status_pause_close_retrain_stop(self):
        s = self.srv
        placed = s.place_trade("EURUSD", "BUY", reason="before the bot")   # MCP trades while no bot
        self.assertTrue(placed["placed"], placed)
        self.config.MCP_TRADING_MODE = "dry_run"
        self.assertIn("MCP_TRADING_MODE", s.start_bot(live=True)["error"])  # [ACT-BC-7]
        self.config.MCP_TRADING_MODE = "live"
        r = s.start_bot(live=False)
        self.assertTrue(r["started"], r)
        st = s.bot_status()
        self.assertTrue(st["bot"]["running"])
        self.assertEqual(st["bot"]["mode"], "demo")
        self.assertIn("already running", s.start_bot()["error"])
        gov_file = [f for f in os.listdir("observability") if f.startswith("governor_state")][0]
        def gov_text():
            with open(os.path.join("observability", gov_file)) as fh:
                return fh.read()
        before = gov_text()
        # one trader per account: the MCP server does not trade or write the governor
        p = s.place_trade("USDJPY", "BUY")
        self.assertFalse(p["placed"])
        self.assertTrue(any("bot.py is running" in b for b in p["blockers"]), p["blockers"])
        self.assertEqual(len(self.fake.orders_sent), 1)
        self.assertEqual(gov_text(), before)
        self.assertTrue(s.pause_trading()["bot_confirmed"])
        self.assertTrue(s.bot_status()["bot"]["paused"])
        c = s.close_position(placed["ticket"])
        self.assertTrue(c["closed"], c)
        self.assertIn("bot.py", c["by"])
        self.assertEqual(s.retrain_ml_model()["n_samples"], 42)
        stop = s.stop_bot()
        self.assertTrue(stop["stopped"], stop)
        self.assertEqual(stop["how"], "graceful")
        self.assertFalse(s.bot_status()["bot"]["running"])
        s.resume_trading()
        self.assertTrue(s.place_trade("USDJPY", "BUY")["placed"], "lock not freed after stop")

    def test_mcp_does_not_book_while_a_bot_holds_the_account(self):
        """INV-BC-3: an MCP trade closes by its stop loss while bot.py runs — only
        the bot may book it (it does, as an orphan); the MCP server leaves it."""
        s = self.srv
        t = s.place_trade("EURUSD", "BUY", reason="x")["ticket"]
        self.fake.stop_out(t, -50.0)
        bot = bc.BotSide("live", account="1")            # this test process plays bot.py
        self.assertTrue(bot.acquire(wait=0)[0])
        try:
            a = s.get_account_status()
            self.assertEqual(a["newly_booked_closed_trades"], [])
            self.assertEqual(s.get_trades_to_review()["newly_booked"], [])
            row = next(r for r in s._journal().get_recent_trades(10) if r["ticket"] == t)
            self.assertEqual(row["status"], "OPEN")
        finally:
            bot.release()
        b = s.get_account_status()["newly_booked_closed_trades"]   # bot gone: MCP books it, once
        self.assertEqual([x["ticket"] for x in b], [t])
        self.assertEqual(s.get_account_status()["newly_booked_closed_trades"], [])

    def test_force_stop_and_crash_free_the_account(self):
        s = self.srv
        with mock.patch.dict(os.environ, {"FAKE_BOT_IGNORE_STOP": "1"}):
            self.assertTrue(s.start_bot()["started"])
        soft = s.stop_bot(force=False, wait_seconds=2)
        self.assertFalse(soft["stopped"])
        self.assertTrue(soft["stop_requested"])
        hard = s.stop_bot(force=True, wait_seconds=1)
        self.assertEqual((hard["stopped"], hard["how"]), (True, "forced"))
        self.assertFalse(bc.bot_state()["running"])
        self.assertTrue(s.start_bot()["started"])                  # a crash: kill -9
        pid = bc.read_status()["pid"]
        self.assertTrue(bc.kill_pid(pid))     # hard kill (SIGKILL / TerminateProcess)
        s._CHILDREN[pid].wait(5)
        self.assertFalse(bc.bot_state()["running"], "OS did not free the lock")
        self.assertTrue(s.place_trade("EURUSD", "BUY")["placed"])

    def test_over_http_only_safe_controls(self):
        s = self.srv
        s._STATE["transport"] = "http"
        try:
            self.assertFalse(s.start_bot()["started"])
            self.assertFalse(s.resume_trading()["resumed"])
            self.assertTrue(s.pause_trading()["paused"])
            self.assertTrue(s.update_settings({"MAX_OPEN_POSITIONS": 1})["saved"])
            self.assertFalse(s.update_settings({"MAX_OPEN_POSITIONS": 9})["saved"])
        finally:
            s._STATE["transport"] = "stdio"

    def test_real_bot_py_refuses_to_start_twice(self):
        holder = bc.BotSide("demo")            # this test process plays the running bot
        self.assertTrue(holder.acquire(wait=0)[0])
        try:
            p = subprocess.run([sys.executable, os.path.join(ROOT, "bot.py"), "--demo"],
                               cwd=self.tmp, capture_output=True, text=True, timeout=120,
                               env=dict(os.environ, PYTHONPATH=os.pathsep.join([FAKE, ROOT])))
            self.assertEqual(p.returncode, 2, p.stdout[-2000:] + p.stderr[-2000:])
            self.assertIn("already running", p.stdout + p.stderr)
        finally:
            holder.release()

    def test_training_job(self):
        s = self.srv
        self.assertFalse(s.run_training("nope")["started"])
        self.assertFalse(s.run_training("tick_csv", path="/no/such/dir")["started"])
        self.assertFalse(s.run_training("mt5_history", symbol="EUR;rm -rf")["started"])
        j = s.run_training("mt5_history", symbol="EURUSD", bars=600, max_samples=20)
        self.assertTrue(j["started"], j)
        end = time.time() + 120
        while s.get_training_job(j["job_id"])["running"] and time.time() < end:
            time.sleep(0.5)
        res = s.get_training_job(j["job_id"])
        self.assertFalse(res["running"])
        self.assertEqual(res["exit_code"], 0, res["output"])


if __name__ == "__main__":
    unittest.main()
