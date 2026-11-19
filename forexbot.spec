# PyInstaller recipe for ForexBot.exe. Build with build_exe.bat (Windows) or
# `python build_app.py`. One folder (fast start; no unpacking on every launch).
#
# config.py is NOT bundled: build_app.py copies it next to the exe, so settings
# stay editable and the app imports it from there (app_paths.setup()).
from PyInstaller.utils.hooks import collect_submodules

hidden = []
for pkg in ("mcp", "uvicorn", "starlette", "sse_starlette", "pydantic_settings",
            "agents", "strategy", "data", "ml", "memory", "observability",
            "optimization", "notifications", "backtest"):
    try:
        hidden += collect_submodules(pkg)
    except Exception:
        pass
hidden += ["bot", "mcp_server", "dashboard.app", "mt5_historical_trainer",
           "csv_historical_trainer", "trainer", "bot_control", "trade_booking",
           "sqlite_conn", "app_paths", "MetaTrader5"]

a = Analysis(
    ["forexbot.py"],
    pathex=["."],
    hiddenimports=hidden,
    datas=[("dashboard/templates", "dashboard/templates")],
    excludes=["config", "tests", "tkinter", "IPython", "jupyter", "notebook", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ForexBot",
    console=True,            # a command-line program: logs show in the window
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ForexBot", upx=False)
