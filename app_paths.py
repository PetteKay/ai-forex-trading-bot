"""
Where the bot lives, whether it runs from source (python bot.py) or as the
packaged Windows app (ForexBot.exe, built with PyInstaller).

  app_dir()   the folder with config.py, .env and every file the bot writes:
              observability/, memory/, data/, bot.log. From source: the repo.
              Packaged: the folder that holds ForexBot.exe.
  bundle_dir()  read-only files shipped inside the app (dashboard templates).
              From source: the repo. Packaged: PyInstaller's bundle folder.
  command(name)  how to start another part of the bot as a child process:
              [python, bot.py, ...] from source, [ForexBot.exe, run, ...] packaged.
  setup()     called once at start-up: work from app_dir(), make config.py
              importable from there, create the data folders, load .env.
"""
from __future__ import annotations

import os
import sys

FROZEN = bool(getattr(sys, "frozen", False))
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# subcommand of ForexBot.exe -> source script
SCRIPTS = {
    "run":          "bot.py",
    "mcp":          "mcp_server.py",
    "dashboard":    os.path.join("dashboard", "app.py"),
    "train-mt5":    "mt5_historical_trainer.py",
    "train-ticks":  "csv_historical_trainer.py",
    "train-trades": "trainer.py",
}


def app_dir() -> str:
    if FROZEN:
        return os.path.dirname(os.path.abspath(sys.executable))
    return _SRC_DIR


def bundle_dir() -> str:
    if FROZEN:
        return getattr(sys, "_MEIPASS", app_dir())
    return _SRC_DIR


def python_exe() -> str:
    """Interpreter for child processes (pythonw.exe would have no console)."""
    exe = sys.executable or "python"
    if os.name == "nt" and exe.lower().endswith("pythonw.exe"):
        cand = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.exists(cand):
            return cand
    return exe


def command(name: str, script: str | None = None) -> list[str]:
    """Argv to start part `name` of the bot (see SCRIPTS) in a new process.
    `script` overrides the source script path (tests use a stand-in bot)."""
    if FROZEN and script is None:
        return [sys.executable, name]
    return [python_exe(), script or os.path.join(_SRC_DIR, SCRIPTS[name])]


def load_dotenv(path: str) -> list[str]:
    """KEY=VALUE lines from a .env file into os.environ. Variables already set
    (e.g. Windows user environment variables) win. Returns the keys loaded.
    No values are logged."""
    loaded = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    except OSError:
        return loaded
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key.replace("_", "").isalnum():
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:                                   # trailing comment
            val = val.split(" #", 1)[0].rstrip()
        if key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


def root_dir() -> str:
    """app_dir(), or FOREX_BOT_DIR if set (tests, or a data folder elsewhere)."""
    return os.environ.get("FOREX_BOT_DIR") or app_dir()


_env_loaded = False


def load_env_once() -> None:
    """Load <root>/.env into os.environ (no side effects besides that).
    config.py calls this first, so LLM_API_KEY etc. can live in .env."""
    global _env_loaded
    if not _env_loaded:
        _env_loaded = True
        load_dotenv(os.path.join(root_dir(), ".env"))


def setup() -> str:
    """Called by each program at start (never on import): work from root_dir(),
    make config.py importable from there, create the data folders, load .env.
    Returns root_dir()."""
    root = root_dir()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)          # the packaged app imports config.py from here
    for d in ("observability", "memory", "data"):
        os.makedirs(d, exist_ok=True)
    load_env_once()
    return root
