"""
Discovered-signal bridge — applies alpha signals found by the signal-discovery
workflow (signal-discovery/) to live bot decisions.

HOW IT CONNECTS
---------------
The discovery workflow saves accepted signals to
  signal-discovery/src/signal_discovery_workflow/output/signal_*.json
with the fields `signal_code` (a self-contained, portable Python module — the
evaluator's documented contract), `selected_signal` (function name) and
`evaluation_metrics` (mean_ic, p_value, ...). Those signals are CROSS-SECTIONAL:
they were validated by ranking all symbols at each timestamp against forward
returns (Rank IC), so the honest way to consume one at trade time is:

  1. build the same OHLCV panel layout the evaluator used
     (rows = H1 timestamps, columns = symbols) from live MT5 candles,
  2. run the saved signal function on the panel,
  3. take the LAST row and z-score it across symbols,
  4. flip the sign when mean_ic < 0 (signal anti-predicts returns),
  5. use each symbol's z-score as a directional tilt:
     positive z -> signal favors this symbol outperforming the panel (BUY tilt),
     negative z -> underperforming (SELL tilt).

Quality gate: a stored signal is only used if its saved metrics pass the same
acceptance bar the workflow used (|IC| and p-value thresholds re-checked here —
never trust file existence alone).
"""

import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("DiscoveredSignals")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "signal-discovery" / "src" / "signal_discovery_workflow" / "output"

_SIGNALS_TTL_S = 3600   # re-scan the output dir at most hourly
_PANEL_TTL_S   = 600    # rebuild the MT5 panel at most every 10 minutes

_loaded_signals: list[dict] | None = None
_loaded_ts: float = 0.0
_panel_scores: dict[str, float] = {}
_panel_ts: float = 0.0


def _passes_quality_gate(metrics: dict, ic_threshold: float, p_threshold: float) -> bool:
    mean_ic = metrics.get("mean_ic")
    p_value = metrics.get("p_value")
    if mean_ic is None or abs(mean_ic) < ic_threshold:
        return False
    if p_value is not None and p_value > p_threshold:
        return False
    return True


def _compile_signal(payload: dict) -> dict | None:
    """Exec the saved portable signal code and return a runnable record."""
    code = payload.get("signal_code", "")
    name = payload.get("selected_signal", "")
    if not code or not name:
        return None
    namespace: dict = {}
    try:
        exec(code, namespace)
    except Exception as e:
        log.warning(f"Discovered signal '{name}' failed to compile: {e}")
        return None
    fn = namespace.get(name)
    if not callable(fn):
        log.warning(f"Discovered signal '{name}' not found in its own code")
        return None
    return {
        "name":    name,
        "fn":      fn,
        "mean_ic": float(payload["evaluation_metrics"]["mean_ic"]),
        "sign":    1.0 if float(payload["evaluation_metrics"]["mean_ic"]) >= 0 else -1.0,
    }


def load_signals(output_dir: str | Path | None = None,
                 ic_threshold: float = 0.015,
                 p_threshold: float = 0.05) -> list[dict]:
    """
    Load, quality-gate, and compile accepted signals from the discovery output.
    Deduped by signal name (best |IC| wins). Cached for _SIGNALS_TTL_S.
    """
    global _loaded_signals, _loaded_ts
    now = time.time()
    if _loaded_signals is not None and (now - _loaded_ts) < _SIGNALS_TTL_S:
        return _loaded_signals

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    best_by_name: dict[str, dict] = {}

    for fp in sorted(out_dir.glob("signal_*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                payload = json.load(f)
            metrics = payload.get("evaluation_metrics", {})
            if not _passes_quality_gate(metrics, ic_threshold, p_threshold):
                continue
            name = payload.get("selected_signal", "")
            prev = best_by_name.get(name)
            if prev is None or abs(metrics["mean_ic"]) > abs(prev["evaluation_metrics"]["mean_ic"]):
                best_by_name[name] = payload
        except Exception as e:
            log.debug(f"Skipping {fp.name}: {e}")

    compiled = []
    for payload in best_by_name.values():
        rec = _compile_signal(payload)
        if rec:
            compiled.append(rec)
            log.info(
                f"Discovered signal loaded: {rec['name']} "
                f"(IC={rec['mean_ic']:+.4f}, direction={'normal' if rec['sign'] > 0 else 'inverted'})"
            )

    _loaded_signals = compiled
    _loaded_ts = now
    if not compiled:
        log.debug("No accepted discovered signals found (or none pass the quality gate)")
    return compiled


def _build_panel(symbols: list[str], mt5_bridge, timeframe: str = "H1", count: int = 250) -> dict[str, pd.DataFrame]:
    """Build the evaluator's panel layout {field: rows=time, cols=symbol} from MT5."""
    frames: dict[str, dict[str, pd.Series]] = {f: {} for f in ("Open", "High", "Low", "Close", "Volume")}
    for sym in symbols:
        try:
            df = mt5_bridge.get_candles(sym, timeframe, count=count)
        except Exception as e:
            log.debug(f"Panel fetch failed for {sym}: {e}")
            continue
        frames["Open"][sym]   = df["open"]
        frames["High"][sym]   = df["high"]
        frames["Low"][sym]    = df["low"]
        frames["Close"][sym]  = df["close"]
        frames["Volume"][sym] = df["volume"]
    return {field: pd.DataFrame(cols) for field, cols in frames.items() if cols}


def compute_scores(symbols: list[str], mt5_bridge, timeframe: str = "H1",
                   output_dir: str | Path | None = None,
                   ic_threshold: float = 0.015,
                   p_threshold: float = 0.05) -> dict[str, float]:
    """
    Return {symbol: composite z-score} across all accepted discovered signals.

    Positive score = signals expect the symbol to OUTPERFORM the panel over the
    next few hours (BUY tilt); negative = underperform (SELL tilt).
    Returns {} when no signals are accepted or the panel can't be built —
    callers treat that as "feature unavailable", never as a neutral vote.
    """
    global _panel_scores, _panel_ts
    now = time.time()
    if _panel_scores and (now - _panel_ts) < _PANEL_TTL_S:
        return _panel_scores

    signals = load_signals(output_dir, ic_threshold, p_threshold)
    if not signals:
        return {}

    panel = _build_panel(symbols, mt5_bridge, timeframe)
    if "Close" not in panel or panel["Close"].shape[1] < 10:
        log.debug("Panel too thin for cross-sectional scoring (need >= 10 symbols)")
        return {}

    import inspect
    from collections import defaultdict

    totals: dict[str, float] = defaultdict(float)
    used = 0
    for rec in signals:
        try:
            sig = inspect.signature(rec["fn"])
            kwargs = {}
            for pname in sig.parameters:
                for field in ("Close", "Volume", "High", "Low", "Open"):
                    pl, fl = pname.lower(), field.lower()
                    if pl == fl or fl in pl or pl in fl:
                        kwargs[pname] = panel[field]
                        break
            if len(kwargs) != len([p for p in sig.parameters.values()
                                   if p.default is inspect.Parameter.empty]):
                # fall back: positional standard fields for unmatched params
                std = [panel[f] for f in ("Close", "Volume", "High", "Low", "Open") if f in panel]
                result = rec["fn"](*std[:len(sig.parameters)])
            else:
                result = rec["fn"](**kwargs)

            if isinstance(result, pd.Series):
                result = result.to_frame().T
            # Symbols' latest bars can end at different timestamps (session
            # closes, fetch hiccups), leaving the literal last row nearly
            # empty. Use the most recent row with enough cross-sectional
            # coverage instead (search back up to 12 rows).
            last = None
            for back in range(1, min(13, len(result) + 1)):
                row = result.iloc[-back].dropna()
                if len(row) >= 10:
                    last = row
                    break
            if last is None:
                continue
            std = float(last.std())
            if std < 1e-12 or math.isnan(std):
                continue
            z = (last - float(last.mean())) / std
            for sym, val in z.items():
                totals[sym] += rec["sign"] * float(val)
            used += 1
        except Exception as e:
            log.debug(f"Discovered signal '{rec['name']}' scoring error: {e}")

    if used == 0:
        return {}

    scores = {sym: round(total / used, 3) for sym, total in totals.items()}
    _panel_scores = scores
    _panel_ts = now
    log.info(f"Discovered signals: {used} active | scores computed for {len(scores)} symbols")
    return scores
