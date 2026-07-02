"""
Economic Calendar — News Blackout System.

Prevents the bot from trading around high-impact news events
(NFP, FOMC, ECB, CPI, GDP, etc.) that cause unpredictable volatility.

Strategy:
  - Block trading N minutes BEFORE and M minutes AFTER each event
  - Built-in recurring events (NFP = first Friday of month, etc.)
  - Optional CSV override for specific scheduled dates
  - Optionally fetch from a free public API (tradingeconomics / myfxbook)

Usage:
  from data.calendar import EconomicCalendar
  cal = EconomicCalendar(blackout_before_min=30, blackout_after_min=60)
  if cal.is_blackout():
      log.info("Blackout active — skipping trade")
"""

import csv
import json
import logging
import os
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

log = logging.getLogger("Calendar")

# ── Live calendar cache ──────────────────────────────────────────────────────
_LIVE_CACHE: dict = {"events": [], "fetched_at": 0.0}
_LIVE_CACHE_TTL  = 3600.0  # 1-hour cache


def _et_to_utc_offset(dt: datetime) -> int:
    """Return hours to ADD to US Eastern time to get UTC. EDT=+4, EST=+5."""
    year = dt.year
    mar1      = datetime(year, 3,  1)
    dst_start = datetime(year, 3,  (7 - mar1.weekday()) % 7 + 8)   # 2nd Sunday of March
    nov1      = datetime(year, 11, 1)
    dst_end   = datetime(year, 11, (7 - nov1.weekday()) % 7 + 1)   # 1st Sunday of November
    return 4 if dst_start.date() <= dt.date() < dst_end.date() else 5


def _parse_ff_event(ev: dict) -> Optional[dict]:
    """Parse one ForexFactory JSON event. Returns None for non-HIGH impact or bad data."""
    if ev.get("impact", "").lower() != "high":
        return None
    date_str = ev.get("date", "")
    time_str = (ev.get("time") or "").strip().lower()
    name     = ev.get("title", "Event")
    try:
        date_naive = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None
    if time_str and time_str not in ("tentative", "all day", ""):
        try:
            t        = datetime.strptime(time_str.upper(), "%I:%M%p")
            et_hour  = t.hour
            et_min   = t.minute
        except ValueError:
            et_hour, et_min = 12, 0
    else:
        et_hour, et_min = 12, 0

    et_naive = date_naive.replace(hour=et_hour, minute=et_min)
    offset   = _et_to_utc_offset(et_naive)
    utc_dt   = (et_naive + timedelta(hours=offset)).replace(tzinfo=timezone.utc)
    return {"name": name, "time": utc_dt, "impact": "HIGH"}


def _fetch_live_events() -> list[dict]:
    """
    Fetch this week's + next week's HIGH-impact events from ForexFactory.
    Cached for 1 hour. Returns [] if unreachable (recurring rules take over).
    """
    now_mono = _time.monotonic()
    if now_mono - _LIVE_CACHE["fetched_at"] < _LIVE_CACHE_TTL and _LIVE_CACHE["events"]:
        return _LIVE_CACHE["events"]

    events: list[dict] = []
    for suffix in ("thisweek", "nextweek"):
        url = f"https://nfs.faireconomy.media/ff_calendar_{suffix}.json"
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (trading-bot/2.0)"})
            with urlopen(req, timeout=8) as resp:
                raw = json.loads(resp.read())
            for ev in raw:
                parsed = _parse_ff_event(ev)
                if parsed:
                    events.append(parsed)
        except URLError as e:
            log.debug(f"Live calendar ({suffix}) unreachable: {e}")
        except Exception as e:
            log.debug(f"Live calendar parse error ({suffix}): {e}")

    if events:
        _LIVE_CACHE["events"]     = events
        _LIVE_CACHE["fetched_at"] = now_mono
        log.info(f"Live calendar: {len(events)} HIGH-impact events loaded from ForexFactory")
    else:
        log.debug("Live calendar fetch failed — falling back to recurring rules")
    return events


# ── Recurring event rules ────────────────────────────────────────────────────
# Each entry: (name, day_of_week, week_of_month, hour_utc, minute_utc, impact)
# day_of_week: 0=Mon … 6=Sun; week_of_month: 1-5 or -1 for last
# These are approximations — CSV overrides take precedence.

RECURRING_EVENTS = [
    # NFP — first Friday of month, 12:30 UTC
    {"name": "NFP",          "dow": 4, "week": 1, "hour": 12, "minute": 30, "impact": "HIGH"},
    # US CPI — roughly 2nd week Wednesday 12:30 (approximate)
    {"name": "US CPI",       "dow": 2, "week": 2, "hour": 12, "minute": 30, "impact": "HIGH"},
    # FOMC — 8 times/year; approximated as 3rd Wednesday 18:00 every other month
    {"name": "FOMC",         "dow": 2, "week": 3, "hour": 18, "minute": 0,  "impact": "HIGH"},
    # ECB — roughly every 6 weeks Thursday 12:15 UTC
    {"name": "ECB Rate",     "dow": 3, "week": 2, "hour": 12, "minute": 15, "impact": "HIGH"},
    # BoE — roughly every 6 weeks Thursday 12:00
    {"name": "BoE Rate",     "dow": 3, "week": 1, "hour": 12, "minute": 0,  "impact": "HIGH"},
    # US GDP — quarterly, last Thursday of Jan/Apr/Jul/Oct, 12:30
    {"name": "US GDP",       "dow": 3, "week": -1,"hour": 12, "minute": 30, "impact": "HIGH"},
]


def _nth_weekday_of_month(year: int, month: int, dow: int, n: int) -> Optional[datetime]:
    """Return the nth (1-based) weekday of a month. n=-1 means last."""
    from calendar import monthrange
    first_day, last_day_num = monthrange(year, month)
    days_in_month = last_day_num

    if n > 0:
        # offset from first occurrence
        first_occur = (dow - first_day) % 7 + 1
        day = first_occur + (n - 1) * 7
        if day > days_in_month:
            return None
        return datetime(year, month, day, tzinfo=timezone.utc)
    else:  # last
        last_occur = days_in_month - (monthrange(year, month)[0] + days_in_month - 1 - dow) % 7
        return datetime(year, month, last_occur, tzinfo=timezone.utc)


def _recurring_events_for_month(year: int, month: int) -> list[dict]:
    events = []
    for rule in RECURRING_EVENTS:
        dt = _nth_weekday_of_month(year, month, rule["dow"], rule["week"])
        if dt is None:
            continue
        dt = dt.replace(hour=rule["hour"], minute=rule["minute"], second=0, microsecond=0)
        events.append({"name": rule["name"], "time": dt, "impact": rule["impact"]})
    return events


class EconomicCalendar:
    def __init__(
        self,
        blackout_before_min: int = 30,
        blackout_after_min:  int = 60,
        csv_path: Optional[str] = None,
        live_enabled: bool = True,
    ):
        self._before       = timedelta(minutes=blackout_before_min)
        self._after        = timedelta(minutes=blackout_after_min)
        self._csv          = csv_path
        self._live_enabled = live_enabled
        self._custom_events: list[dict] = []
        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)

    def is_blackout(self, now: Optional[datetime] = None) -> tuple[bool, str]:
        """
        Returns (True, reason) if currently in a news blackout window.
        Returns (False, "") if safe to trade.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        all_events = self._get_events_around(now)
        for event in all_events:
            et = event["time"]
            if et.tzinfo is None:
                et = et.replace(tzinfo=timezone.utc)
            if (et - self._before) <= now <= (et + self._after):
                minutes_to = int((et - now).total_seconds() / 60)
                if minutes_to >= 0:
                    reason = f"BLACKOUT: {event['name']} in {minutes_to}m"
                else:
                    reason = f"BLACKOUT: {event['name']} ended {-minutes_to}m ago"
                log.info(reason)
                return True, reason

        return False, ""

    def next_event(self, now: Optional[datetime] = None) -> Optional[dict]:
        """Return the next upcoming high-impact event."""
        if now is None:
            now = datetime.now(timezone.utc)
        events = self._get_events_around(now)
        future = [e for e in events if e["time"] > now]
        if not future:
            return None
        return min(future, key=lambda e: e["time"])

    def _get_events_around(self, now: datetime) -> list[dict]:
        """Collect events from recurring rules + CSV overrides + live API (if enabled)."""
        events = []
        for delta_month in (-1, 0, 1):
            y, m = divmod(now.month - 1 + delta_month, 12)
            events.extend(_recurring_events_for_month(now.year + y, m + 1))
        events.extend(self._custom_events)
        if self._live_enabled:
            live = _fetch_live_events()
            if live:
                # Live events override recurring when names overlap (live data is precise)
                recurring_names_lower = {e["name"].lower() for e in events}
                for ev in live:
                    if ev["name"].lower() not in recurring_names_lower:
                        events.append(ev)
                    else:
                        # Replace the approximate recurring entry with the exact live time
                        events = [e for e in events if e["name"].lower() != ev["name"].lower()]
                        events.append(ev)
        return events

    def _load_csv(self, path: str):
        """
        Load custom events from CSV.
        Expected columns: name, datetime (ISO format), impact
        Example row: NFP Special,2026-07-04T12:30:00,HIGH
        """
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        dt = datetime.fromisoformat(row["datetime"].strip())
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        self._custom_events.append({
                            "name":   row.get("name", "Custom Event"),
                            "time":   dt,
                            "impact": row.get("impact", "HIGH"),
                        })
                    except (ValueError, KeyError):
                        continue
            log.info(f"Loaded {len(self._custom_events)} custom calendar events from {path}")
        except Exception as e:
            log.warning(f"Could not load calendar CSV {path}: {e}")
