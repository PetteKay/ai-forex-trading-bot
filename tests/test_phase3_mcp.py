"""
Phase 3: MCP server tests, against a fake MT5 terminal (tests/fake_mt5).

  • In-process: every tool through a real MCP client session.
  • stdio:      mcp_server.py started as a subprocess, exactly as Claude Code /
                Claude Desktop / Codex launch it.
  • HTTP:       the streamable-HTTP app behind the token check, as ChatGPT uses it.

Run: py -m pytest tests/test_phase3_mcp.py -v
"""
import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import anyio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = os.path.join(ROOT, "tests", "fake_mt5")
sys.path.insert(0, ROOT)
logging.disable(logging.CRITICAL)

import numpy as np    # noqa: E402,F401
import pandas as pd   # noqa: E402,F401


def _load_fake_mt5():
    spec = importlib.util.spec_from_file_location("MetaTrader5", os.path.join(FAKE, "MetaTrader5.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _result(r):
    """Structured dict from a CallToolResult."""
    if getattr(r, "structured_content", None):
        return r.structured_content
    for c in r.content:
        if getattr(c, "text", None):
            try:
                return json.loads(c.text)
            except Exception:
                return {"_text": c.text}
    return {}


class MCPInProcess(unittest.TestCase):
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
        self.patches = [mock.patch.object(config, k, v, create=True) for k, v in dict(
            LIVE_CALENDAR_ENABLED=False, CALENDAR_CSV_PATH=os.path.join(self.tmp, "none.csv"),
            ALLOWED_SESSIONS=[], SPREAD_FILTER_ENABLED=False, SIZING_MODE="risk_pct",
            RISK_PER_TRADE_PCT=0.5, MCP_TRADING_MODE="dry_run", MCP_HTTP_ALLOW_TRADING=False,
            MAX_OPEN_POSITIONS=5, MAX_ORDERS_PER_DAY=10).items()]
        for p in self.patches:
            p.start()
        self.config = config
        import mcp_server
        self.srv = mcp_server
        sys.modules["data.macro_context"] = type(sys)("data.macro_context")
        sys.modules["data.macro_context"].get_macro_context = lambda s: {}
        import data.news_feed as nf
        self.nf_patch = mock.patch.object(nf, "fetch_headlines", return_value=["Dollar firm"])
        self.nf_patch.start()

    def tearDown(self):
        self.nf_patch.stop()
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

    def call(self, name, **args):
        from mcp.client import Client

        async def go():
            async with Client(self.srv.server) as c:
                return _result(await c.call_tool(name, args))
        return anyio.run(go)

    def list_tools(self):
        from mcp.client import Client

        async def go():
            async with Client(self.srv.server) as c:
                return (await c.list_tools()).tools
        return anyio.run(go)

    # ── tools exist, with read-only / destructive hints ──────────────────
    def test_tools_and_annotations(self):
        tools = {t.name: t for t in self.list_tools()}
        expected = {"get_account_status", "analyze_market", "get_news", "get_ml_probability",
                    "recall_lessons", "get_recent_trades", "preview_trade", "place_trade",
                    "close_position"}
        self.assertEqual(set(tools), expected)
        for n in ("place_trade", "close_position"):
            self.assertTrue(tools[n].annotations.destructive_hint)
        for n in expected - {"place_trade", "close_position"}:
            self.assertTrue(tools[n].annotations.read_only_hint)

    def test_account_and_market(self):
        acct = self.call("get_account_status")
        self.assertEqual(acct["equity"], 10_000.0)
        self.assertEqual(acct["trading_mode"], "dry_run")
        m = self.call("analyze_market", symbol="EURUSD")
        self.assertEqual(m["symbol"], "EURUSD")
        self.assertIn("regime", m)
        self.assertIn("atr", m)
        news = self.call("get_news", symbol="EURUSD")
        self.assertEqual(news["headlines"], ["Dollar firm"])
        ml = self.call("get_ml_probability", symbol="EURUSD", direction="BUY")
        self.assertTrue(0.0 <= ml["probability"] <= 1.0)

    def test_preview_sizes_from_risk(self):
        p = self.call("preview_trade", symbol="EURUSD", direction="BUY")
        self.assertTrue(p["allowed"], p["blockers"])
        self.assertGreater(p["lot"], 0)
        self.assertLessEqual(p["risk_usd_if_stopped"], 10_000 * 0.005 + 1e-6)
        self.assertLess(p["stop_loss"], p["approx_entry"])
        self.assertGreater(p["take_profit"], p["approx_entry"])

    def test_dry_run_sends_no_order(self):
        r = self.call("place_trade", symbol="EURUSD", direction="BUY")
        self.assertFalse(r["placed"])
        self.assertTrue(r["dry_run"])
        self.assertEqual(self.fake.orders_sent, [])

    def test_live_order_goes_through_governor_and_guards(self):
        self.config.MCP_TRADING_MODE = "live"
        r = self.call("place_trade", symbol="EURUSD", direction="BUY", reason="test")
        self.assertTrue(r["placed"], r)
        self.assertEqual(len(self.fake.orders_sent), 1)
        with open(self.config.GOVERNOR_STATE_PATH) as f:
            self.assertEqual(json.load(f)["orders_today"], 1)
        dup = self.call("place_trade", symbol="EURUSD", direction="BUY")
        self.assertFalse(dup["placed"])
        self.assertTrue(any("Already have" in b for b in dup["blockers"]))
        corr = self.call("place_trade", symbol="GBPUSD", direction="BUY")
        self.assertFalse(corr["placed"])
        self.assertTrue(any("Correlation" in b for b in corr["blockers"]))
        self.assertEqual(len(self.fake.orders_sent), 1)
        closed = self.call("close_position", ticket=r["ticket"])
        self.assertTrue(closed["closed"])

    def test_kill_switch_from_bot_state_blocks_trading(self):
        self.config.MCP_TRADING_MODE = "live"
        from datetime import datetime, timezone
        os.makedirs("observability", exist_ok=True)
        with open(self.config.GOVERNOR_STATE_PATH, "w") as f:
            json.dump({"version": 1, "today": datetime.now(timezone.utc).date().isoformat(),
                       "daily_pnl": "-300", "kill_active": True, "orders_today": 2,
                       "day_anchor": "10000", "drawdown_halt": False, "peak_equity": "10000"}, f)
        r = self.call("place_trade", symbol="EURUSD", direction="BUY")
        self.assertFalse(r["placed"])
        self.assertTrue(any("Kill switch" in b for b in r["blockers"]), r["blockers"])
        self.assertEqual(self.fake.orders_sent, [])

    def test_trading_off_and_http_without_permission(self):
        self.config.MCP_TRADING_MODE = "off"
        self.assertIn("off", self.call("place_trade", symbol="EURUSD", direction="BUY")["error"])
        self.config.MCP_TRADING_MODE = "live"
        self.srv._STATE["transport"] = "http"
        try:
            r = self.call("place_trade", symbol="EURUSD", direction="BUY")
            self.assertFalse(r["placed"])
            self.assertIn("HTTP", r["error"])
            self.assertEqual(self.fake.orders_sent, [])
            self.assertIn("equity", self.call("get_account_status"))   # read-only still works
        finally:
            self.srv._STATE["transport"] = "stdio"

    def test_rejects_bad_arguments(self):
        r = self.call("preview_trade", symbol="EURUSD", direction="HOLD")
        self.assertNotIn("allowed", r)       # tool error, no plan


    def test_simultaneous_orders_cannot_exceed_max_positions(self):
        self.config.MCP_TRADING_MODE = "live"
        self.config.MAX_OPEN_POSITIONS = 1
        results = []

        def one(sym):
            results.append(self.srv.place_trade(sym, "BUY"))
        ts = [threading.Thread(target=one, args=(s,)) for s in ("EURUSD", "USDJPY", "AUDUSD")]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(sum(1 for r in results if r.get("placed")), 1, results)
        self.assertEqual(len(self.fake.orders_sent), 1)


class MCPStdio(unittest.TestCase):
    """Starts mcp_server.py exactly as an MCP client would (stdin/stdout)."""

    def test_stdio_session(self):
        from mcp.client import Client
        from mcp.client.stdio import StdioServerParameters
        tmp = tempfile.mkdtemp()
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([FAKE, ROOT]), FOREX_BOT_DIR=tmp)
        params = StdioServerParameters(command=sys.executable,
                                       args=[os.path.join(ROOT, "mcp_server.py")],
                                       env=env, cwd=os.path.expanduser("~"))

        async def go():
            async with Client(params, read_timeout_seconds=60) as c:
                names = {t.name for t in (await c.list_tools()).tools}
                acct = _result(await c.call_tool("get_account_status", {}))
                mkt = _result(await c.call_tool("analyze_market", {"symbol": "EURUSD"}))
                return names, acct, mkt
        names, acct, mkt = anyio.run(go)
        self.assertTrue(os.path.exists(os.path.join(tmp, "observability", "governor_state.json")),
                        "server did not work from FOREX_BOT_DIR")
        self.assertIn("place_trade", names)
        self.assertEqual(acct["equity"], 10_000.0)
        self.assertEqual(mkt["symbol"], "EURUSD")


class MCPHttpAuth(unittest.TestCase):
    """Streamable HTTP behind the token check (the ChatGPT path)."""

    @classmethod
    def setUpClass(cls):
        import socket
        import uvicorn
        sys.modules["MetaTrader5"] = _load_fake_mt5()
        for m in ("mt5_bridge", "mcp_server"):
            sys.modules.pop(m, None)
        import mcp_server
        cls.token = "t" * 32
        cls.tmp = tempfile.mkdtemp()
        cls.old = os.getcwd()
        os.chdir(cls.tmp)
        s = socket.socket(); s.bind(("127.0.0.1", 0)); cls.port = s.getsockname()[1]; s.close()
        app = mcp_server.build_http_app(cls.token, ["tunnel.example.com"])
        cls.uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=cls.port, log_level="error"))
        cls.th = threading.Thread(target=cls.uv.run, daemon=True)
        cls.th.start()
        for _ in range(100):
            if cls.uv.started:
                break
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.uv.should_exit = True
        cls.th.join(timeout=5)
        os.chdir(cls.old)
        for m in ("mt5_bridge", "mcp_server", "MetaTrader5"):
            sys.modules.pop(m, None)

    def _post(self, url, headers=None):
        """POST an MCP initialize request; returns an object with .status_code."""
        import types
        import urllib.error
        import urllib.request
        body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "t", "version": "1"}}}
        h = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
        h.update(headers or {})
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read(1)
                return types.SimpleNamespace(status_code=r.status)
        except urllib.error.HTTPError as e:
            return types.SimpleNamespace(status_code=e.code)

    def test_no_or_wrong_token_is_rejected(self):
        base = f"http://127.0.0.1:{self.port}/mcp"
        self.assertEqual(self._post(base).status_code, 401)
        self.assertEqual(self._post(base + "?token=wrong").status_code, 401)
        self.assertEqual(self._post(base, {"authorization": "Bearer wrong"}).status_code, 401)

    def test_token_in_header_or_url_is_accepted(self):
        base = f"http://127.0.0.1:{self.port}/mcp"
        self.assertEqual(self._post(base, {"authorization": f"Bearer {self.token}"}).status_code, 200)
        self.assertEqual(self._post(f"{base}?token={self.token}").status_code, 200)

    def test_unknown_host_header_is_rejected(self):
        base = f"http://127.0.0.1:{self.port}/mcp?token={self.token}"
        self.assertNotEqual(self._post(base, {"host": "evil.example.org"}).status_code, 200)
        self.assertEqual(self._post(base, {"host": "tunnel.example.com"}).status_code, 200)

    def test_full_mcp_session_over_http(self):
        from mcp.client import Client
        url = f"http://127.0.0.1:{self.port}/mcp?token={self.token}"

        async def go():
            async with Client(url, read_timeout_seconds=30) as c:
                return {t.name for t in (await c.list_tools()).tools}
        self.assertIn("analyze_market", anyio.run(go))


if __name__ == "__main__":
    unittest.main(verbosity=2)

