"""
The packaged app (ForexBot.exe) and the code paths it depends on.

  * app_paths: .env loading, the bot's folder, how child processes are started
    (python bot.py from source / ForexBot.exe run when packaged)
  * config.py reads keys from .env; Windows environment variables win
  * forexbot.py launcher: help, version, unknown command, every command's --help
    (each imports its whole program, so a missing import shows up here)
  * dashboard/app.py starts from any folder (regression: it could not import
    the bot's modules when run as `py dashboard/app.py`)

The build itself (PyInstaller, forexbot.spec) is checked by build_exe.bat's
smoke test and the release workflow.

Run: py -m pytest tests/test_app_packaging.py -v
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = os.path.join(ROOT, "tests", "fake_mt5")
sys.path.insert(0, ROOT)

import app_paths  # noqa: E402


def _run(args, cwd, **env):
    e = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"}   # only what the test sets
    e.update(PYTHONPATH=os.pathsep.join([FAKE, ROOT]), **env)
    return subprocess.run([sys.executable] + args, cwd=cwd, env=e, capture_output=True,
                          text=True, timeout=180)


class DotEnv(unittest.TestCase):
    def test_parsing_and_existing_variables_win(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, ".env")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\ufeff# comment\nLLM_API_KEY=abc123\nexport TELEGRAM_BOT_TOKEN='tok en'\n"
                    "PLAIN = value # note\nQUOTED=\"a#b\"\nbad line\n1BAD-KEY=x\nALREADY=new\n")
        env = {"ALREADY": "old"}
        with mock.patch.dict(os.environ, env, clear=True):
            loaded = app_paths.load_dotenv(p)
            self.assertEqual(os.environ["LLM_API_KEY"], "abc123")
            self.assertEqual(os.environ["TELEGRAM_BOT_TOKEN"], "tok en")
            self.assertEqual(os.environ["PLAIN"], "value")
            self.assertEqual(os.environ["QUOTED"], "a#b")
            self.assertEqual(os.environ["ALREADY"], "old")
            self.assertNotIn("ALREADY", loaded)
        self.assertEqual(app_paths.load_dotenv(os.path.join(d, "missing.env")), [])

    def test_config_reads_dotenv_and_the_environment_wins(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, ".env"), "w") as f:
            f.write("LLM_API_KEY=from-dotenv\n")
        code = ["-c", "import config; print(config.LLM_API_KEY)"]
        r = _run(code, d, FOREX_BOT_DIR=d)
        self.assertEqual(r.stdout.strip(), "from-dotenv", r.stderr)
        r = _run(code, d, FOREX_BOT_DIR=d, LLM_API_KEY="from-windows-env")
        self.assertEqual(r.stdout.strip(), "from-windows-env", r.stderr)


class ChildCommands(unittest.TestCase):
    def test_from_source(self):
        self.assertEqual(app_paths.command("run")[1], os.path.join(ROOT, "bot.py"))
        self.assertEqual(app_paths.command("train-mt5")[1:], [os.path.join(ROOT, "mt5_historical_trainer.py")])
        self.assertEqual(app_paths.command("run", "/x/fake_bot.py")[1:], ["/x/fake_bot.py"])

    def test_packaged(self):
        with mock.patch.object(app_paths, "FROZEN", True), \
             mock.patch.object(sys, "executable", r"C:\ForexBot\ForexBot.exe"):
            self.assertEqual(app_paths.command("run"), [r"C:\ForexBot\ForexBot.exe", "run"])
            self.assertEqual(app_paths.command("train-ticks"), [r"C:\ForexBot\ForexBot.exe", "train-ticks"])

    def test_every_program_exists(self):
        for name, script in app_paths.SCRIPTS.items():
            self.assertTrue(os.path.isfile(os.path.join(ROOT, script)), name)

    def test_root_dir_and_setup(self):
        d = tempfile.mkdtemp()
        old = os.getcwd()
        try:
            with mock.patch.dict(os.environ, {"FOREX_BOT_DIR": d}):
                self.assertEqual(app_paths.setup(), d)
                self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(d))
                for sub in ("observability", "memory", "data"):
                    self.assertTrue(os.path.isdir(os.path.join(d, sub)))
                self.assertIn(d, sys.path)
        finally:
            os.chdir(old)
            if d in sys.path:
                sys.path.remove(d)


class Launcher(unittest.TestCase):
    def test_help_version_unknown(self):
        d = tempfile.mkdtemp()
        r = _run([os.path.join(ROOT, "forexbot.py")], d, FOREX_BOT_DIR=d)
        self.assertEqual(r.returncode, 0)
        self.assertIn("ForexBot run", r.stdout)
        r = _run([os.path.join(ROOT, "forexbot.py"), "version"], d, FOREX_BOT_DIR=d)
        self.assertEqual(r.stdout.strip(), "ForexBot 1.0.0")
        r = _run([os.path.join(ROOT, "forexbot.py"), "nope"], d, FOREX_BOT_DIR=d)
        self.assertEqual(r.returncode, 2)

    def test_every_command_loads(self):
        """`<command> --help` imports the whole program behind it."""
        d = tempfile.mkdtemp()
        for name in app_paths.SCRIPTS:
            r = _run([os.path.join(ROOT, "forexbot.py"), name, "--help"], d, FOREX_BOT_DIR=d)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr[-1500:]}")
            self.assertIn("usage:", r.stdout.lower(), name)

    def test_dashboard_runs_as_a_script_from_any_folder(self):
        d = tempfile.mkdtemp()
        e = dict(os.environ, FOREX_BOT_DIR=d)
        e.pop("PYTHONPATH", None)                 # exactly like `py dashboard/app.py`
        r = subprocess.run([sys.executable, os.path.join(ROOT, "dashboard", "app.py"), "--help"],
                           cwd=d, env=e, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])

    def test_dashboard_serves_from_the_bots_folder(self):
        d = tempfile.mkdtemp()
        code = ("import os,sys; sys.path.insert(0, %r); "
                "from dashboard import app as a; c = a.app.test_client(); "
                "r = c.get('/'); s = c.get('/api/stats'); "
                "print(r.status_code, s.status_code, a._DB_PATH.startswith(%r))") % (ROOT, d)
        r = _run(["-c", code], d, FOREX_BOT_DIR=d)
        self.assertEqual(r.stdout.split(), ["200", "200", "True"], r.stderr[-1500:])


if __name__ == "__main__":
    unittest.main()


class ApiKeyCheck(unittest.TestCase):
    def test_placeholder_or_empty_key_is_a_config_error(self):
        import types
        from agents import base
        for key in ("", "your-api-key-here", "your-nvapi-key-here"):
            cfg = types.SimpleNamespace(LLM_PROVIDER="nvidia", LLM_API_KEY=key,
                                        LLM_MODEL="m", LLM_BASE_URL=None)
            with self.assertRaises(base.ConfigError, msg=key):
                base.init(cfg)

    def test_claude_may_use_anthropic_api_key(self):
        import types
        from agents import base
        cfg = types.SimpleNamespace(LLM_PROVIDER="claude", LLM_API_KEY="", LLM_MODEL="m", LLM_BASE_URL=None)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            try:
                base.init(cfg)
            except ImportError:
                self.skipTest("anthropic not installed")


class Supervisor(unittest.TestCase):
    """bot.py's restart supervisor (found by the packaged-app smoke test)."""

    def test_missing_api_key_stops_with_a_message_instead_of_a_restart_loop(self):
        d = tempfile.mkdtemp()
        r = _run([os.path.join(ROOT, "bot.py"), "--demo"], d, FOREX_BOT_DIR=d)
        self.assertEqual(r.returncode, 2, (r.stdout + r.stderr)[-2000:])
        out = r.stdout + r.stderr
        self.assertIn("LLM_API_KEY is not set", out)
        self.assertIn(".env", out)
        self.assertNotIn("restarting in", out)

    def test_restart_wait_honours_a_stop_request(self):
        import importlib.util
        import threading
        import time
        spec = importlib.util.spec_from_file_location("MetaTrader5", os.path.join(FAKE, "MetaTrader5.py"))
        fake = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fake)
        had = sys.modules.get("MetaTrader5")
        sys.modules["MetaTrader5"] = fake
        for m in ("bot", "mt5_bridge", "position_manager"):
            sys.modules.pop(m, None)
        d, old = tempfile.mkdtemp(), os.getcwd()
        try:
            with mock.patch.dict(os.environ, {"FOREX_BOT_DIR": d}):
                import bot
                import bot_control as bc
                calls = []

                def crash(**kw):
                    calls.append(1)
                    raise ValueError("boom")
                result = {}
                with mock.patch.object(bot, "run", crash), mock.patch.object(bot, "_RESTART_DELAY_S", 60):
                    t = threading.Thread(target=lambda: result.update(rc=bot.main(["--demo"])))
                    t.start()
                    st = None
                    for _ in range(100):
                        st = bc.read_status(os.path.join(d, bc.STATUS_PATH))
                        if calls and st.get("instance"):
                            break
                        time.sleep(0.05)
                    c = bc.set_paused(True, os.path.join(d, bc.CONTROL_PATH))
                    ok = bc.wait_for(lambda s: s.get("control_version", -1) >= c["version"], 10,
                                     os.path.join(d, bc.STATUS_PATH))
                    self.assertIsNotNone(ok, "pause not confirmed during the restart wait")
                    t0 = time.monotonic()
                    bc.update_control(lambda c: c["stop"].append(st["instance"]),
                                      os.path.join(d, bc.CONTROL_PATH), os.path.join(d, bc.STATUS_PATH))
                    t.join(15)
                    self.assertFalse(t.is_alive(), "stop ignored during the restart wait")
                    self.assertLess(time.monotonic() - t0, 10)
                    self.assertEqual(result.get("rc"), 0)
                    self.assertEqual(len(calls), 1)
        finally:
            os.chdir(old)
            for m in ("bot", "mt5_bridge", "position_manager"):
                sys.modules.pop(m, None)
            if had is not None:
                sys.modules["MetaTrader5"] = had
            else:
                sys.modules.pop("MetaTrader5", None)
