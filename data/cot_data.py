"""
CFTC Commitment of Traders (COT) Data — Institutional Positioning Signal.

Downloads the weekly disaggregated COT report from CFTC (free, no API key).
Caches each weekly report in SQLite to accumulate history for z-score computation.

Signal: Large Speculator Net Position (Long - Short)
  Positive z-score → institutionals are unusually long → bullish momentum
  Negative z-score → institutionals are unusually short → bearish momentum
  |z-score| > 1.5 → strong directional conviction

COT is released every Friday for prior Tuesday's snapshot.
This module fetches once per week and uses SQLite cache otherwise.
"""

import csv
import io
import logging
import os
import sqlite3
import time
from urllib.request import urlopen, Request
from urllib.error import URLError

log = logging.getLogger("COTData")

_CFTC_URL    = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
_CACHE_TTL_S = 7 * 24 * 3600   # 7 days

# CME contract name fragments → (our_symbol, invert?)
# invert=True: net long at CME = net short our symbol (e.g., JPY long = USDJPY short)
_CONTRACT_MAP: dict[str, tuple[str, bool]] = {
    "EURO FX":           ("EURUSD", False),
    "BRITISH POUND":     ("GBPUSD", False),
    "JAPANESE YEN":      ("USDJPY", True),
    "AUSTRALIAN DOLLAR": ("AUDUSD", False),
    "NEW ZEALAND DOLLAR":("NZDUSD", False),
    "CANADIAN DOLLAR":   ("USDCAD", True),
    "SWISS FRANC":       ("USDCHF", True),
    "GOLD":              ("XAUUSD", False),
}

_last_fetch: float = 0.0
_cache: dict[str, dict] = {}


def _init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""
            CREATE TABLE IF NOT EXISTS cot_weekly (
                symbol      TEXT NOT NULL,
                report_date TEXT NOT NULL,
                net_long    INTEGER DEFAULT 0,
                longs       INTEGER DEFAULT 0,
                shorts      INTEGER DEFAULT 0,
                PRIMARY KEY (symbol, report_date)
            )
        """)


def _parse_cot_csv(text: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            market = row.get("Market and Exchange Names", "").upper()
            date_s = row.get("As of Date in Form YYYY-MM-DD", "").strip()
            try:
                longs  = int(row.get("Noncommercial Positions-Long (All)", 0) or 0)
                shorts = int(row.get("Noncommercial Positions-Short (All)", 0) or 0)
            except (ValueError, TypeError):
                continue
            for key, (symbol, inverted) in _CONTRACT_MAP.items():
                if key in market:
                    net = (shorts - longs) if inverted else (longs - shorts)
                    result[symbol] = {"date": date_s, "net_long": net, "longs": longs, "shorts": shorts}
                    break
    except Exception as e:
        log.warning(f"COT parse error: {e}")
    return result


def _fetch_and_store(db_path: str) -> dict[str, dict]:
    try:
        req  = Request(_CFTC_URL, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=30)
        data = _parse_cot_csv(resp.read().decode("utf-8", errors="replace"))
        if not data:
            log.warning("COT: 0 rows parsed — CFTC format may have changed")
            return {}
        with sqlite3.connect(db_path) as con:
            for sym, d in data.items():
                con.execute("""
                    INSERT OR REPLACE INTO cot_weekly (symbol, report_date, net_long, longs, shorts)
                    VALUES (?, ?, ?, ?, ?)
                """, (sym, d["date"], d["net_long"], d["longs"], d["shorts"]))
        log.info(f"COT: fetched {len(data)} contracts")
        return data
    except URLError as e:
        log.warning(f"COT: CFTC unreachable ({e}) — using cache")
        return {}
    except Exception as e:
        log.warning(f"COT: fetch error: {e}")
        return {}


def _load_from_db(db_path: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    try:
        with sqlite3.connect(db_path) as con:
            rows = con.execute("""
                SELECT symbol, report_date, net_long, longs, shorts
                FROM cot_weekly c
                WHERE report_date = (
                    SELECT MAX(report_date) FROM cot_weekly c2 WHERE c2.symbol = c.symbol
                )
            """).fetchall()
        for sym, date_s, net, longs, shorts in rows:
            result[sym] = {"date": date_s, "net_long": net, "longs": longs, "shorts": shorts}
    except Exception:
        pass
    return result


def _compute_zscore(symbol: str, current_net: int, db_path: str, weeks: int = 26) -> float:
    try:
        with sqlite3.connect(db_path) as con:
            rows = con.execute("""
                SELECT net_long FROM cot_weekly WHERE symbol=?
                ORDER BY report_date DESC LIMIT ?
            """, (symbol, weeks)).fetchall()
        if len(rows) < 4:
            return 0.0
        vals  = [r[0] for r in rows]
        mean  = sum(vals) / len(vals)
        std   = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        return round((current_net - mean) / std, 2) if std > 1 else 0.0
    except Exception:
        return 0.0


def get_cot_signal(symbol: str, db_path: str = "data/cot_history.db") -> dict:
    """
    Return current COT institutional positioning signal for a symbol.

    Returns:
      net_long:   int   — net large speculator position
      z_score:    float — how extreme vs 26-week history
      signal:     str   — BULLISH_INSTITUTIONAL | BEARISH_INSTITUTIONAL | NEUTRAL
      available:  bool  — False when no COT data exists for this symbol
    """
    global _last_fetch, _cache

    sym = symbol.upper().split(".")[0]

    if time.monotonic() - _last_fetch > _CACHE_TTL_S:
        _init_db(db_path)
        fresh = _fetch_and_store(db_path)
        _cache = fresh if fresh else _load_from_db(db_path)
        _last_fetch = time.monotonic()

    if sym not in _cache:
        return {"symbol": sym, "net_long": 0, "z_score": 0.0, "signal": "NEUTRAL", "date": "", "available": False}

    d   = _cache[sym]
    net = d["net_long"]
    z   = _compute_zscore(sym, net, db_path)

    # Require both z-score AND absolute threshold to avoid low-position false signals
    if z > 1.0 and net > 50_000:
        signal = "BULLISH_INSTITUTIONAL"
    elif z < -1.0 and net < -50_000:
        signal = "BEARISH_INSTITUTIONAL"
    else:
        signal = "NEUTRAL"

    return {
        "symbol":    sym,
        "net_long":  net,
        "z_score":   z,
        "signal":    signal,
        "date":      d["date"],
        "available": True,
    }
