"""
Bot control — how an MCP client (Claude Code, Claude Desktop, Codex, ChatGPT ...)
controls bot.py, and how the two share one trading account safely.

Formal model: specs/BotControl.tla (TLC-checked). The names in brackets below are
the spec's actions and invariants this code implements.

  observability/bot.lock          OS file lock on the account. Held by a running
                                  bot.py for its whole life, or by the MCP server
                                  while it places, closes or books trades. The OS
                                  frees it when the holder dies.        [lock]
  observability/bot_status.json   written by bot.py: instance id, pid, heartbeat,
                                  and what it last read from the control file.
  observability/bot_control.json  written by MCP tools: pause flag, stop requests
                                  (by instance id), close and retrain requests
                                  (by instance id).                     [control]
  observability/settings_overrides.json
                                  AI-set limits; may only TIGHTEN config.py.

Guarantees (TLC-checked in the spec; tested in tests/test_bot_control.py):
  INV-BC-1/2  one trader per account: a running bot holds the lock; the MCP
              server trades only while it holds it.
  INV-BC-3    each closed trade is booked once: only the lock holder books.
  INV-BC-4    a stop request names an instance; a later bot ignores it.
  INV-BC-5    the bot confirms a pause only between orders (main thread), so
              after a confirmed pause it has nothing in flight and sends nothing.
  INV-BC-6    effective limit = the tighter of config.py and the override, so
              a later config.py edit still caps it.
  ACT-BC-7    a live bot starts only if config.MCP_TRADING_MODE == "live".
  ACT-BC-8    nothing here clears the governor's kill switch.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Optional

log = logging.getLogger("BotControl")

LOCK_PATH      = os.path.join("observability", "bot.lock")
STATUS_PATH    = os.path.join("observability", "bot_status.json")
CONTROL_PATH   = os.path.join("observability", "bot_control.json")
OVERRIDES_PATH = os.path.join("observability", "settings_overrides.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Files ────────────────────────────────────────────────────────────────────

def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_json(path: str, data: dict) -> None:
    """Atomic: readers never see a half-written file."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=str)
    for attempt in range(20):            # Windows: replace fails while a reader has it open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


# ── OS file locks ────────────────────────────────────────────────────────────

class _FileLock:
    """Exclusive, non-blocking OS lock on one file. Freed by the OS if we die."""

    def __init__(self, path: str):
        self.path = path
        self._fh = None

    def try_acquire(self) -> bool:
        if self._fh is not None:
            return True
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._fh.close()
        self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire_blocking(self, timeout: float = 10.0) -> bool:
        end = time.monotonic() + timeout
        while not self.try_acquire():
            if time.monotonic() > end:
                return False
            time.sleep(0.02)
        return True


class _ReentrantAccountLock:
    """The account lock as used inside ONE process (the MCP server): threads
    queue on a threading.RLock, and the OS lock is taken once and kept while
    any nested `hold()` is active. [McpAcquire / McpRelease]"""

    def __init__(self, path: str):
        self._os = _FileLock(path)
        self._rlock = threading.RLock()
        self._depth = 0

    @contextmanager
    def hold(self):
        """Yields True if this process holds the account lock for the block,
        False if another process (a running bot.py) has it."""
        with self._rlock:
            if self._depth == 0 and not self._os.try_acquire():
                yield False
                return
            self._depth += 1
            try:
                yield True
            finally:
                self._depth -= 1
                if self._depth == 0:
                    self._os.release()

    @property
    def held_here(self) -> bool:
        return self._os.held


_account_locks: dict[str, _ReentrantAccountLock] = {}


def account_lock(path: str = LOCK_PATH) -> _ReentrantAccountLock:
    path = os.path.abspath(path)
    if path not in _account_locks:
        _account_locks[path] = _ReentrantAccountLock(path)
    return _account_locks[path]


# ── Processes ────────────────────────────────────────────────────────────────

def pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except Exception:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                     # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid(pid: int) -> bool:
    """Force-terminate (stop_bot force=True). [ForceKill]"""
    try:
        import psutil
        p = psutil.Process(int(pid))
        p.kill()
        p.wait(timeout=10)
        return True
    except ImportError:
        pass
    except Exception:
        return not pid_alive(pid)
    if os.name == "nt":
        import subprocess
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.5)
    return not pid_alive(pid)


def read_status(path: str = STATUS_PATH) -> dict:
    return _read_json(path)


def bot_state(lock_path: str = LOCK_PATH, status_path: str = STATUS_PATH) -> dict:
    """Is a bot.py running on this account?

    running = the account lock is held by another process AND the status file
    names a live process. (The lock alone could be another MCP server for a
    moment; the status alone could be stale after a crash.)
    """
    st = read_status(status_path)
    lk = account_lock(lock_path)
    if lk.held_here:
        held_elsewhere = False
    else:
        probe = _FileLock(os.path.abspath(lock_path))
        held_elsewhere = not probe.try_acquire()
        probe.release()
    alive = bool(st.get("pid")) and pid_alive(st.get("pid"))
    running = held_elsewhere and alive and st.get("state") in ("starting", "running", "stopping")
    return {"running": running, "lock_held_elsewhere": held_elsewhere, "status": st,
            "instance": st.get("instance") if running else None}


# ── Control file (MCP side writes) ───────────────────────────────────────────

def read_control(path: str = CONTROL_PATH) -> dict:
    c = _read_json(path)
    c.setdefault("version", 0)
    c.setdefault("paused", False)
    c.setdefault("stop", [])
    c.setdefault("close", [])
    c.setdefault("retrain", {})
    return c


def update_control(fn: Callable[[dict], None], path: str = CONTROL_PATH,
                   status_path: str = STATUS_PATH) -> dict:
    """Read-modify-write under a writer lock (several MCP servers may write)."""
    wl = _FileLock(os.path.abspath(path) + ".lock")
    if not wl.acquire_blocking(timeout=10):
        raise TimeoutError("control file is locked by another writer")
    try:
        c = read_control(path)
        fn(c)
        c["version"] = int(c.get("version", 0)) + 1
        c["updated"] = _now()
        # requests for instances that are no longer running are dead: keep the file small
        live = {read_status(status_path).get("instance")}
        c["stop"] = [i for i in c["stop"] if i in live][-20:]
        c["close"] = [r for r in c["close"] if r and r[0] in live][-50:]
        c["retrain"] = {k: v for k, v in c["retrain"].items() if k in live}
        _write_json(path, c)
        return c
    finally:
        wl.release()


def set_paused(paused: bool, path: str = CONTROL_PATH) -> dict:          # [SetPause]
    def f(c):
        c["paused"] = bool(paused)
    return update_control(f, path)


def request_stop(instance: str, path: str = CONTROL_PATH) -> dict:      # [RequestStop]
    def f(c):
        if instance not in c["stop"]:
            c["stop"].append(instance)
    return update_control(f, path)


def request_close(instance: str, ticket: int, path: str = CONTROL_PATH) -> dict:   # [RequestClose]
    def f(c):
        r = [instance, int(ticket)]
        if r not in c["close"]:
            c["close"].append(r)
    return update_control(f, path)


def request_retrain(instance: str, path: str = CONTROL_PATH) -> tuple[dict, int]:
    seq = {}

    def f(c):
        seq["n"] = int(c["retrain"].get(instance, 0)) + 1
        c["retrain"][instance] = seq["n"]
    c = update_control(f, path)
    return c, seq["n"]


def wait_for(pred: Callable[[dict], bool], timeout: float, status_path: str = STATUS_PATH,
             interval: float = 0.25) -> Optional[dict]:
    """Poll bot_status.json until pred(status) or timeout. Returns the status or None."""
    end = time.monotonic() + timeout
    while True:
        st = read_status(status_path)
        if st and pred(st):
            return st
        if time.monotonic() > end:
            return None
        time.sleep(interval)


# ── Bot side ─────────────────────────────────────────────────────────────────

class BotSide:
    """Used by bot.py: holds the account lock for the process lifetime, writes
    the status heartbeat and reads the control file.

    poll() must only be called from bot.py's main thread and never while an
    order is being sent — that is what makes a confirmed pause safe
    (spec: BotReadControl requires ~inFlight; mutation M4 shows the violation).
    """

    def __init__(self, mode: str, account: str = "", lock_path: str = LOCK_PATH,
                 status_path: str = STATUS_PATH, control_path: str = CONTROL_PATH):
        self.instance = uuid.uuid4().hex[:12]
        self.mode = mode
        self.account = account
        self._lock = _FileLock(os.path.abspath(lock_path))
        self._status_path = os.path.abspath(status_path)     # fixed even if cwd changes
        self._control_path = os.path.abspath(control_path)
        self._status: dict = {}
        self._done_closes: dict = {}
        self._retrain_done = 0

    # [Launch -> Acquire | LoseRace]
    def acquire(self, wait: float = 30.0) -> tuple[bool, str]:
        end = time.monotonic() + wait
        while True:
            if self._lock.try_acquire():
                self._status = {"instance": self.instance, "pid": os.getpid(), "mode": self.mode,
                                "account": self.account, "state": "starting", "started": _now(),
                                "heartbeat": _now(), "control_version": -1, "paused": None,
                                "closes": {}, "retrain_done": 0, "tick": 0}
                _write_json(self._status_path, self._status)
                return True, ""
            st = read_status(self._status_path)
            other = st.get("pid")
            if (other and int(other) != os.getpid() and pid_alive(other)
                    and st.get("state") in ("starting", "running", "stopping")):
                return False, (f"another bot.py is already running on this account "
                               f"(pid {other}, instance {st.get('instance')})")
            if time.monotonic() > end:     # held by an MCP server for too long
                return False, "the account lock stayed busy (an MCP server is trading?)"
            time.sleep(0.25)

    def update(self, **fields) -> None:
        self._status.update(fields)
        self._status["heartbeat"] = _now()
        _write_json(self._status_path, self._status)

    def poll(self) -> dict:
        """[BotReadControl] Read control, confirm what was read, return the
        commands addressed to THIS instance."""
        c = read_control(self._control_path)
        stop = self.instance in c.get("stop", [])
        closes = [int(t) for inst, t in c.get("close", []) if inst == self.instance
                  and str(int(t)) not in self._done_closes]
        want = int(c.get("retrain", {}).get(self.instance, 0))
        self._status.update({"control_version": int(c.get("version", 0)),
                             "paused": bool(c.get("paused", False)),
                             "state": "stopping" if stop else "running"})
        self.update()
        return {"paused": bool(c.get("paused", False)), "stop": stop, "close": closes,
                "retrain": want if want > self._retrain_done else 0}

    def report_close(self, ticket: int, result: str) -> None:
        self._done_closes[str(int(ticket))] = result
        self.update(closes=dict(self._done_closes))

    def report_retrain(self, seq: int, result: dict) -> None:
        self._retrain_done = seq
        self.update(retrain_done=seq, retrain_result=result)

    def release(self, state: str = "stopped") -> None:
        try:
            self.update(state=state, stopped=_now())
        finally:
            self._lock.release()


# ── Settings the AI may change: tighten only ─────────────────────────────────
# kind: "max"  smaller is safer   (0 < v <= config.py value)
#       "min"  larger is safer    (config.py value <= v <= upper bound)
#       "subset"                  (non-empty subset of config.py list)
TIGHTEN_ONLY: dict[str, tuple] = {
    "RISK_PER_TRADE_PCT":      ("max", float),
    "MAX_LOT_SIZE":            ("max", float),
    "MAX_OPEN_POSITIONS":      ("max", int),
    "MAX_ORDERS_PER_DAY":      ("max", int),
    "DAILY_LOSS_CAP_PCT":      ("max", float),
    "MAX_DRAWDOWN_PCT":        ("max", float),
    "META_LABEL_MIN_PROB":     ("min", float, 0.95),
    "MIN_CONFIDENCE_TO_TRADE": ("min", int, 100),
    "SYMBOLS":                 ("subset", list),
}


def _file_values(cfg) -> dict:
    """config.py's own values. The current attribute is config.py's value unless
    an override of ours changed it and it still holds what we set (cfg._APPLIED);
    so anything set by someone else since — or equal to config.py — is taken as
    the human's value."""
    import copy
    fv = dict(getattr(cfg, "_FILE_VALUES", {}) or {})
    applied = getattr(cfg, "_APPLIED", {}) or {}
    for k in TIGHTEN_ONLY:
        cur = getattr(cfg, k, None)
        ours = k in fv and k in applied and applied[k][1] and cur == applied[k][0]
        if not ours:
            fv[k] = copy.deepcopy(list(cur) if isinstance(cur, tuple) else cur)
    cfg._FILE_VALUES = fv
    return fv


def validate(key: str, value, file_value):
    """Returns (ok, normalised value or error). [SetAiLimit: v <= humanLimit]"""
    if key not in TIGHTEN_ONLY:
        return False, (f"{key} can't be changed from an MCP client. Allowed: "
                       f"{', '.join(TIGHTEN_ONLY)}. Trading mode, keys and loosening any "
                       f"limit are config.py-only.")
    kind, typ = TIGHTEN_ONLY[key][0], TIGHTEN_ONLY[key][1]
    if kind == "subset":
        if not isinstance(value, (list, tuple)) or not value:
            return False, f"{key} must be a non-empty list"
        allowed = list(file_value or [])
        bad = [s for s in value if s not in allowed]
        if bad:
            return False, f"{key}: {bad} not in config.py's list {allowed}"
        return True, [s for s in allowed if s in value]
    try:
        v = typ(value)
    except (TypeError, ValueError):
        return False, f"{key} must be a {typ.__name__}"
    if isinstance(value, float) and typ is int and value != int(value):
        return False, f"{key} must be a whole number"
    if v != v or v in (float("inf"), float("-inf")):
        return False, f"{key} must be a finite number"
    if kind == "max":
        if v <= 0:
            return False, f"{key} must be > 0"
        if file_value is not None and v > file_value:
            return False, (f"{key} = {v} would loosen config.py's {file_value}. "
                           f"From MCP it can only be lowered.")
        return True, v
    upper = TIGHTEN_ONLY[key][2]
    if v > upper:
        return False, f"{key} must be <= {upper}"
    if file_value is not None and v < file_value:
        return False, (f"{key} = {v} would loosen config.py's {file_value}. "
                       f"From MCP it can only be raised.")
    return True, v


def effective(key: str, file_value, override):
    """[EffectiveLimit] The tighter of config.py and the override — so lowering
    config.py later still caps an older override (mutation M3 shows why)."""
    if override is None:
        return file_value
    kind = TIGHTEN_ONLY[key][0]
    if file_value is None:
        return override
    if kind == "max":
        return min(file_value, override)
    if kind == "min":
        return max(file_value, override)
    return [s for s in file_value if s in override] or list(file_value)


def read_overrides(path: str = OVERRIDES_PATH) -> dict:
    return {k: v for k, v in _read_json(path).items() if k in TIGHTEN_ONLY}


def set_overrides(cfg, changes: dict, path: str = OVERRIDES_PATH) -> dict:
    """Validate all, then write all (nothing is written if any is rejected)."""
    fv = _file_values(cfg)
    errors, ok = {}, {}
    for k, v in (changes or {}).items():
        good, res = validate(k, v, fv.get(k))
        (ok if good else errors)[k] = res
    if errors:
        return {"saved": False, "errors": errors}
    wl = _FileLock(os.path.abspath(path) + ".lock")
    if not wl.acquire_blocking(timeout=10):
        return {"saved": False, "errors": {"_": "settings file is busy"}}
    try:
        cur = read_overrides(path)
        cur.update(ok)
        _write_json(path, cur)
    finally:
        wl.release()
    return {"saved": True, "overrides": cur,
            "effective": {k: effective(k, fv.get(k), cur.get(k)) for k in cur}}


def clear_overrides(keys: Optional[list] = None, path: str = OVERRIDES_PATH) -> dict:   # [ClearAiLimit]
    wl = _FileLock(os.path.abspath(path) + ".lock")
    if not wl.acquire_blocking(timeout=10):
        return {"cleared": False, "error": "settings file is busy"}
    try:
        cur = read_overrides(path)
        for k in (list(cur) if not keys else keys):
            cur.pop(k, None)
        _write_json(path, cur)
    finally:
        wl.release()
    return {"cleared": True, "overrides": cur}


def _sanitize(key: str, value):
    """Type/range check of a stored override, ignoring config.py (None if bad)."""
    kind = TIGHTEN_ONLY[key][0]
    if kind == "subset":
        return [str(x) for x in value] if isinstance(value, list) and value else None
    good, v = validate(key, value, None)
    return v if good else None


def apply_overrides(cfg, path: str = OVERRIDES_PATH) -> dict:
    """Set every tighten-only cfg attribute to its effective value: the tighter
    of config.py and the stored override (config.py's value when there is no
    override, so clearing one restores it). Returns {key: value} for the keys
    that differ from config.py."""
    fv = _file_values(cfg)
    stored = read_overrides(path)
    changed, applied = {}, {}
    for k in TIGHTEN_ONLY:
        if fv.get(k) is None and k not in stored:
            continue
        val = effective(k, fv.get(k), _sanitize(k, stored[k]) if k in stored else None)
        setattr(cfg, k, val)
        applied[k] = (val, val != fv.get(k))      # (value we set, did an override change it)
        if val != fv.get(k):
            changed[k] = val
    cfg._APPLIED = applied
    return changed


def python_exe() -> str:
    """Interpreter for child processes (pythonw.exe would have no console)."""
    exe = sys.executable or "python"
    if os.name == "nt" and exe.lower().endswith("pythonw.exe"):
        cand = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.exists(cand):
            return cand
    return exe


class NullControl:
    """run() without a control file (tests, backtests): never paused or stopped."""
    instance = None

    def poll(self) -> dict:
        return {"paused": False, "stop": False, "close": [], "retrain": 0}

    def update(self, **fields) -> None:
        pass

    def report_close(self, ticket, result) -> None:
        pass

    def report_retrain(self, seq, result) -> None:
        pass

    def release(self, state: str = "stopped") -> None:
        pass
