"""
Stand-in for bot.py in process-level tests: the same start-up lock and control
loop (bot_control.BotSide), without MT5 or LLM calls. Exits 2 if another bot
holds the account, like bot.py.

  FAKE_BOT_IGNORE_STOP=1   never honours a stop request (to test stop_bot force=True)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bot_control  # noqa: E402

mode = "demo" if "--demo" in sys.argv else "live"
side = bot_control.BotSide(mode=mode, account="1")
ok, why = side.acquire(wait=float(os.environ.get("FAKE_BOT_LOCK_WAIT", "5")))
if not ok:
    print(f"Not starting: {why}.", flush=True)
    sys.exit(2)
print(f"fake bot {side.instance} pid {os.getpid()} mode {mode}", flush=True)
ignore_stop = os.environ.get("FAKE_BOT_IGNORE_STOP") == "1"
tick = 0
try:
    side.update(state="running")
    while True:
        cmd = side.poll()
        for t in cmd["close"]:
            side.report_close(t, "closed")
        if cmd["retrain"]:
            side.report_retrain(cmd["retrain"], {"trained": True, "n_samples": 42})
        if cmd["stop"] and not ignore_stop:
            print("stop requested", flush=True)
            break
        tick += 1
        side.update(tick=tick)
        time.sleep(0.1)
finally:
    side.release()
