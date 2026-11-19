"""
Build the packaged app: dist/ForexBot/ (ForexBot.exe + config.py + docs) and
dist/ForexBot-<version>-<platform>.zip with its SHA-256.

    python build_app.py                 build, smoke-test, zip
    python build_app.py --fake-mt5      also for a machine without MetaTrader5
                                        (Linux / CI): the smoke test uses the
                                        test suite's fake terminal

The smoke test runs the BUILT app: `version`, then `<command> --help` for every
command (each imports its whole program, so a module PyInstaller missed fails
here, not on your trading PC).
"""
import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import app_paths  # noqa: E402
from forexbot import VERSION  # noqa: E402

SHIP = ["config.py", ".env.example", "README.md", "SETUP.md", "MCP.md", "LICENSE"]
START_HERE = """ForexBot {version}
================

1. Open config.py (Notepad is fine) and choose your settings and LLM provider.
2. Put your API key in a file named .env in this folder (copy .env.example),
   or set LLM_API_KEY as a Windows environment variable.
3. Open MetaTrader 5 and log in to your account.
4. In this folder, open a terminal (File Explorer: type cmd in the address bar) and run:

     ForexBot run --demo        the bot, analysing and logging only (start here)
     ForexBot run               the bot, trading for real
     ForexBot dashboard         dashboard at http://127.0.0.1:5000
     ForexBot mcp               MCP server for Claude / Codex / ChatGPT (see MCP.md)
     ForexBot train-mt5         pre-train the ML gate from MT5 history
     ForexBot --help            everything else

Everything the bot writes (journal, memory, logs) is kept in this folder.
Trading involves risk of loss. Run in demo first.
"""


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def smoke_test(app_folder: str, fake_mt5: bool) -> None:
    exe = os.path.join(app_folder, "ForexBot.exe" if os.name == "nt" else "ForexBot")
    work = tempfile.mkdtemp(prefix="forexbot-smoke-")
    shutil.copy(os.path.join(app_folder, "config.py"), work)
    if fake_mt5:
        shutil.copy(os.path.join(ROOT, "tests", "fake_mt5", "MetaTrader5.py"), work)
    env = dict(os.environ, FOREX_BOT_DIR=work)
    env.pop("LLM_API_KEY", None)
    out = subprocess.run([exe, "version"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and VERSION in out.stdout, out.stdout + out.stderr
    for name in app_paths.SCRIPTS:
        r = subprocess.run([exe, name, "--help"], env=env, capture_output=True, text=True,
                           timeout=300)
        if r.returncode != 0 or "usage" not in r.stdout.lower():
            raise SystemExit(f"Smoke test failed: ForexBot {name} --help\n{r.stdout}\n{r.stderr}")
        print(f"  ok  ForexBot {name} --help")
    shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake-mt5", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    shutil.rmtree(os.path.join("build", "ForexBot"), ignore_errors=True)
    shutil.rmtree(os.path.join("dist", "ForexBot"), ignore_errors=True)
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "forexbot.spec"])

    folder = os.path.join("dist", "ForexBot")
    for f in SHIP:
        if os.path.exists(f):
            shutil.copy(f, folder)
    with open(os.path.join(folder, "START_HERE.txt"), "w", encoding="utf-8") as fh:
        fh.write(START_HERE.format(version=VERSION))

    print("Smoke test of the built app:")
    smoke_test(os.path.abspath(folder), args.fake_mt5)

    if args.no_zip:
        return 0
    plat = "windows-x64" if os.name == "nt" else f"{platform.system().lower()}-{platform.machine().lower()}"
    zpath = os.path.join("dist", f"ForexBot-{VERSION}-{plat}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, _, files in os.walk(folder):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, os.path.join("ForexBot", os.path.relpath(full, folder)))
    h = hashlib.sha256()
    with open(zpath, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    with open(zpath + ".sha256", "w") as fh:
        fh.write(f"{h.hexdigest()}  {os.path.basename(zpath)}\n")
    size = os.path.getsize(zpath) / 1e6
    print(f"\nBuilt {zpath} ({size:.0f} MB)\nSHA-256 {h.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
