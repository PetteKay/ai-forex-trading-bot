"""
Economic Surprise Index — Actual vs. Expected Macro Release Tracker.

Parses ForexFactory HIGH-impact events that have both actual and forecast values.
Computes a rolling 4-week "surprise score" per currency:
  positive score = recent releases have beaten expectations (bullish)
  negative score = recent releases have missed expectations (bearish)

For a symbol pair (e.g., EURUSD), the net surprise = base surprise - quote surprise.
  Positive net = base currency (EUR) has been beating expectations → bullish for pair.

Data comes from the same ForexFactory source as the economic calendar (no new deps).
Cached in SQLite, refreshed every hour.
"""

import json
import logging
import os
from sqlite_conn import connect as _sqlite_connect
import time
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError

log = logging.getLogger("EconSurprise")

_FF_URL      = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_CACHE_TTL_S = 3600.0
_last_fetch: float = 0.0

# Event title keywords → currency code
_TITLE_TO_CCY: dict[str, str] = {
    "NON-FARM": "USD", "NFP": "USD", "FOMC": "USD", "FED FUNDS": "USD",
    "CONSUMER PRICE INDEX": "USD", "CPI": "USD", "PCE": "USD", "GDP": "USD",
    "JOBLESS": "USD", "UNEMPLOYMENT CLAIMS": "USD", "RETAIL SALES": "USD",
    "ISM": "USD", "PMI": "USD",
    "BANK OF ENGLAND": "GBP", "BOE": "GBP", "CLAIMANT": "GBP",
    "UK CPI": "GBP", "SERVICES PMI": "GBP",
    "ECB": "EUR", "EUROPEAN CENTRAL BANK": "EUR", "EURO ZONE CPI": "EUR",
    "BANK OF JAPAN": "JPY", "BOJ": "JPY",
    "RESERVE BANK OF AUSTRALIA": "AUD", "RBA": "AUD", "AUSTRALIA CPI": "AUD",
    "RESERVE BANK OF NEW ZEALAND": "NZD", "RBNZ": "NZD",
    "BANK OF CANADA": "CAD", "BOC": "CAD", "CANADA CPI": "CAD",
    "SWISS NATIONAL BANK": "CHF", "SNB": "CHF",
}

# ForexFactory "currency" field directly tells us the currency
# But some events lack it, so we also try title matching above.


def _init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with _sqlite_connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""
            CREATE TABLE IF NOT EXISTS econ_surprises (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date  TEXT NOT NULL,
                event_name  TEXT NOT NULL,
                currency    TEXT NOT NULL,
                forecast    REAL,
                actual      REAL,
                surprise    REAL,
                stored_at   TEXT,
                UNIQUE(event_date, event_name, currency)
            )
        """)


def _parse_number(s: str | None) -> float | None:
    if not s or not s.strip():
        return None
    s = s.strip().replace(",", "")
    mult = 1.0
    if s.upper().endswith("K"):
        mult, s = 1_000.0, s[:-1]
    elif s.upper().endswith("M"):
        mult, s = 1_000_000.0, s[:-1]
    elif s.upper().endswith("B"):
        mult, s = 1_000_000_000.0, s[:-1]
    s = s.rstrip("%")
    try:
        return float(s) * mult
    except ValueError:
        return None


def _resolve_currency(ev: dict) -> str:
    ccy = (ev.get("currency") or "").upper().strip()
    if ccy and len(ccy) == 3:
        return ccy
    title = ev.get("title", "").upper()
    for keyword, resolved in _TITLE_TO_CCY.items():
        if keyword in title:
            return resolved
    return ""


def _fetch_and_store(db_path: str) -> None:
    try:
        req    = Request(_FF_URL, headers={"User-Agent": "Mozilla/5.0"})
        resp   = urlopen(req, timeout=20)
        events = json.loads(resp.read().decode("utf-8", errors="replace"))
        now_s  = datetime.now(timezone.utc).isoformat()
        stored = 0
        with _sqlite_connect(db_path) as con:
            for ev in events:
                if ev.get("impact", "").lower() not in ("high", "3"):
                    continue
                actual_s = ev.get("actual", "")
                fc_s     = ev.get("forecast", "")
                if not actual_s or not fc_s:
                    continue
                actual = _parse_number(actual_s)
                fc     = _parse_number(fc_s)
                if actual is None or fc is None:
                    continue

                ccy    = _resolve_currency(ev)
                if not ccy:
                    continue
                date_s   = ev.get("date", "")
                title    = ev.get("title", "")
                surprise = round(((actual - fc) / abs(fc) * 100) if fc != 0 else (actual - fc), 2)

                try:
                    con.execute("""
                        INSERT OR IGNORE INTO econ_surprises
                            (event_date, event_name, currency, forecast, actual, surprise, stored_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (date_s, title, ccy, fc, actual, surprise, now_s))
                    stored += 1
                except Exception:
                    pass
        if stored:
            log.info(f"EconSurprise: stored {stored} surprise events")
    except URLError:
        log.debug("EconSurprise: ForexFactory unreachable — using cache")
    except Exception as e:
        log.debug(f"EconSurprise: fetch error: {e}")


def get_surprise_score(
    currency: str,
    db_path: str  = "data/econ_surprise.db",
    lookback_days: int = 28,
) -> dict:
    """
    Rolling 4-week economic surprise score for a currency.

    Positive score = recent releases beat expectations (bullish currency impulse).
    """
    global _last_fetch

    _init_db(db_path)
    if time.monotonic() - _last_fetch > _CACHE_TTL_S:
        _fetch_and_store(db_path)
        _last_fetch = time.monotonic()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    ccy    = currency.upper()
    try:
        with _sqlite_connect(db_path) as con:
            rows = con.execute("""
                SELECT surprise FROM econ_surprises
                WHERE currency=? AND event_date>=? AND surprise IS NOT NULL
                ORDER BY event_date DESC LIMIT 20
            """, (ccy, cutoff)).fetchall()
    except Exception:
        rows = []

    if not rows:
        return {"score": 0.0, "n_events": 0, "signal": "NEUTRAL", "currency": ccy}

    score = round(sum(r[0] for r in rows), 2)
    signal = (
        "POSITIVE_SURPRISE" if score > 15
        else "NEGATIVE_SURPRISE" if score < -15
        else "NEUTRAL"
    )
    return {"score": score, "n_events": len(rows), "signal": signal, "currency": ccy}


def get_symbol_surprise(symbol: str, db_path: str = "data/econ_surprise.db") -> dict:
    """
    Net surprise alignment for a forex pair.
    Positive → base currency has been beating expectations (bullish for pair).
    """
    sym = symbol.upper().split(".")[0]
    if len(sym) != 6:
        return {"net_surprise": 0.0, "signal": "NEUTRAL", "symbol": sym}

    base_r  = get_surprise_score(sym[:3], db_path)
    quote_r = get_surprise_score(sym[3:], db_path)
    net     = round(base_r["score"] - quote_r["score"], 2)

    signal = (
        "BASE_BULLISH"  if net > 10
        else "BASE_BEARISH" if net < -10
        else "NEUTRAL"
    )
    return {
        "symbol":       sym,
        "net_surprise": net,
        "base_score":   base_r["score"],
        "quote_score":  quote_r["score"],
        "signal":       signal,
    }
