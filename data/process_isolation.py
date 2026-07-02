"""
Process-level isolation for flaky, crash-prone external fetches.

WHY THIS EXISTS
---------------
yfinance's default HTTP backend (curl_cffi) has documented Windows access-violation
crashes that trigger on network failures (DNS errors, resets, timeouts). An access
violation kills the whole interpreter — a Python try/except cannot catch it, because
it isn't a Python exception. If that call happens inline in the live trading loop,
one bad DNS lookup takes the entire bot down with no traceback.

Running the risky call in a child process contains the blast radius: a crash there
can only kill the child. The parent (the live trading bot) always survives and gets
`default` back, exactly like any other best-effort fetch failure.
"""

import logging

log = logging.getLogger("ProcessIsolation")

_CTX = None


def _ctx():
    global _CTX
    if _CTX is None:
        import multiprocessing as mp
        _CTX = mp.get_context("spawn")
    return _CTX


def _worker(func, args, kwargs, queue):
    try:
        queue.put(("ok", func(*args, **kwargs)))
    except Exception as e:
        queue.put(("error", str(e)))


def run_isolated(func, *args, timeout: float = 25.0, default=None, **kwargs):
    """
    Run func(*args, **kwargs) in a child process and return its result.

    Returns `default` if the child times out, crashes (e.g. native access
    violation), or raises a Python exception — the caller never needs its own
    try/except for this call; failure modes all collapse to `default`.

    `func` must be a module-level function (picklable) — not a lambda or
    closure — since the spawn start method pickles it to hand to the child.
    """
    queue = _ctx().Queue()
    proc = _ctx().Process(target=_worker, args=(func, args, kwargs, queue), daemon=True)
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        log.warning(f"{func.__name__} timed out after {timeout}s in isolated process — terminating")
        proc.terminate()
        proc.join(5)
        return default

    if proc.exitcode != 0:
        log.warning(
            f"{func.__name__} crashed in isolated process (exit code {proc.exitcode}) — "
            f"using default instead of taking down the caller"
        )
        return default

    try:
        status, payload = queue.get_nowait()
    except Exception:
        log.warning(f"{func.__name__} produced no result in isolated process — using default")
        return default

    if status == "ok":
        return payload

    log.warning(f"{func.__name__} raised in isolated process: {payload}")
    return default
