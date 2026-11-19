"""
ForexBot — one program for every part of the bot. This is what ForexBot.exe runs;
from source, `python forexbot.py <command>` does the same.

    ForexBot run [--demo]            the trading bot (bot.py)
    ForexBot mcp [--http ...]        MCP server for Claude / Codex / ChatGPT (MCP.md)
    ForexBot dashboard [--port N]    web dashboard on http://127.0.0.1:5000
    ForexBot train-mt5 [...]         ML samples from MT5 price history
    ForexBot train-ticks [...]       ML samples from MT5 tick-export CSVs
    ForexBot train-trades [...]      lessons from a CSV of your past trades
    ForexBot version

Settings: config.py next to ForexBot.exe. API keys: Windows environment
variables, or a .env file next to ForexBot.exe (see .env.example).
"""
from __future__ import annotations

import os
import sys

import app_paths

VERSION = "1.0.0"

# In every process of the packaged app — including multiprocessing "spawn"
# children, which never run main() — config.py next to the exe is importable.
if app_paths.FROZEN and app_paths.root_dir() not in sys.path:
    sys.path.insert(0, app_paths.root_dir())

_HELP = __doc__.split("\n", 1)[1].split("Settings:", 1)[0].rstrip() + """

Add --help after a command for its options, e.g.  ForexBot run --help
Folder: {folder}
"""


def _help() -> str:
    return "ForexBot " + VERSION + "\n" + _HELP.replace("{folder}", app_paths.root_dir())


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_help())
        if not argv and app_paths.FROZEN and sys.stdin and sys.stdin.isatty():
            input("Press Enter to close...")          # double-clicked ForexBot.exe
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in ("version", "--version", "-V"):
        print(f"ForexBot {VERSION}")
        return 0
    if cmd not in app_paths.SCRIPTS:
        print(f"Unknown command: {cmd}\n\n{_help()}", file=sys.stderr)
        return 2

    app_paths.setup()            # work from the bot's folder, load .env, find config.py
    if app_paths.FROZEN and not os.path.exists(os.path.join(app_paths.root_dir(), "config.py")):
        print(f"config.py is missing from {app_paths.root_dir()}. "
              "Copy it from the release zip next to ForexBot.exe.", file=sys.stderr)
        return 2

    if cmd == "run":
        import bot
        return bot.main(rest)
    if cmd == "mcp":
        import mcp_server
        return mcp_server.main(rest) or 0
    if cmd == "dashboard":
        from dashboard import app as dashboard_app
        return dashboard_app.main(rest)
    if cmd == "train-mt5":
        import mt5_historical_trainer
        return mt5_historical_trainer.main(rest)
    if cmd == "train-ticks":
        import csv_historical_trainer
        return csv_historical_trainer.main(rest)
    if cmd == "train-trades":
        import trainer
        return trainer.main(rest)
    return 2


if __name__ == "__main__":
    if app_paths.FROZEN:
        import multiprocessing
        multiprocessing.freeze_support()   # the bot uses a process pool on Windows
    sys.exit(main())
