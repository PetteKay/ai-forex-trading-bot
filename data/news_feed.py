"""
Real-time forex news fetcher via free RSS feeds.
No API key required. Caches results for 5 minutes so the bot
never hammers a feed on every 60-second tick.
"""

import logging
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

log = logging.getLogger("NewsFeed")

# Symbol → keywords for headline relevance scoring
_SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "EURUSD":  ["euro", "eur", "ecb", "european central bank", "fed", "federal reserve",
                "dollar", "usd", "inflation", "cpi", "nfp", "payrolls"],
    "GBPUSD":  ["pound", "gbp", "boe", "bank of england", "sterling", "uk", "britain",
                "fed", "dollar", "usd"],
    "USDJPY":  ["yen", "jpy", "boj", "bank of japan", "japan", "dollar", "fed", "yields"],
    "XAUUSD":  ["gold", "xau", "safe haven", "real yields", "inflation", "fed", "dollar",
                "geopolit", "war", "treasury"],
    "USOIL":   ["oil", "crude", "opec", "wti", "brent", "energy", "barrel"],
    "USDCHF":  ["franc", "chf", "snb", "swiss", "safe haven", "dollar"],
    "AUDUSD":  ["aussie", "aud", "rba", "reserve bank australia", "china", "commodities", "iron"],
    "USDCAD":  ["cad", "canada", "boc", "bank of canada", "oil", "loonie"],
    "NZDUSD":  ["nzd", "new zealand", "rbnz", "kiwi"],
    "EURGBP":  ["euro", "eur", "ecb", "pound", "gbp", "boe", "uk", "european"],
    "NAS100":  ["nasdaq", "tech", "fed", "rate", "inflation", "earnings", "ai"],
    "US30":    ["dow", "fed", "rate", "inflation", "earnings", "industrial"],
    "SPX500":  ["s&p", "spx", "fed", "rate", "inflation", "earnings", "equity"],
}

# Free RSS feeds — no auth, no rate limits
#
# Reuters: feeds.reuters.com stopped updating in June 2020 and no longer
# connects, and reuters.com has no public RSS feed (its arc/outboundfeeds
# addresses return errors). reuters.com is a web page, not a feed this
# parser can read. A Google News search feed limited to site:reuters.com
# returns the latest Reuters stories as standard RSS 2.0 (checked working
# by several feed directories in Sep–Oct 2026).
# To check every feed from your own machine:  py -m data.news_feed
_RSS_FEEDS = [
    "https://www.forexlive.com/feed/news",
    "https://news.google.com/rss/search?q=when:24h+site:reuters.com&ceid=US:en&hl=en-US&gl=US",
    "https://www.marketwatch.com/rss/marketpulse",
]

_CACHE_TTL = timedelta(minutes=5)
_cache_ts: datetime | None = None
_cache_headlines: list[str] = []
# Agent desks for several symbols run on worker threads; when the cache is
# stale only one of them fetches, the others wait and reuse the result.
_fetch_lock = threading.Lock()
# After every feed fails, wait this long before trying again (instead of
# retrying all feeds — up to 6 s each — for every symbol on every tick).
_FAIL_BACKOFF = timedelta(minutes=2)
_fail_ts: datetime | None = None


def fetch_headlines(symbol: str, max_headlines: int = 8) -> list[str]:
    """
    Return recent news headlines most relevant to `symbol`.
    Cached for 5 minutes — safe to call on every bot tick.

    Returns an empty list on total feed failure so the bot continues normally.
    """
    global _cache_ts, _cache_headlines

    now = datetime.now(timezone.utc)
    if _cache_ts and (now - _cache_ts) < _CACHE_TTL:
        return _rank_for_symbol(_cache_headlines, symbol, max_headlines)

    with _fetch_lock:
        now = datetime.now(timezone.utc)
        if _cache_ts and (now - _cache_ts) < _CACHE_TTL:   # filled while we waited
            return _rank_for_symbol(_cache_headlines, symbol, max_headlines)
        if _fail_ts and (now - _fail_ts) < _FAIL_BACKOFF:   # all feeds just failed
            return _rank_for_symbol(_cache_headlines, symbol, max_headlines)
        _refresh(now)
    return _rank_for_symbol(_cache_headlines, symbol, max_headlines)


def _refresh(now: datetime) -> None:
    global _cache_ts, _cache_headlines, _fail_ts
    fresh: list[str] = []
    for url in _RSS_FEEDS:
        try:
            batch = _parse_rss(url)
            fresh.extend(batch)
            log.debug(f"NewsFeed: {len(batch)} headlines from {url}")
        except (URLError, HTTPError) as e:
            log.warning(f"NewsFeed fetch failed ({url}): {e}")
        except ET.ParseError as e:
            log.warning(f"NewsFeed XML parse error ({url}): {e}")
        except Exception as e:
            log.warning(f"NewsFeed unexpected error ({url}): {e}")

    if fresh:
        _cache_ts = now
        _cache_headlines = fresh
        log.info(f"NewsFeed: cached {len(fresh)} headlines total")
        _fail_ts = None
    else:
        _fail_ts = now
        log.warning("NewsFeed: all feeds failed — news analyst will run without live headlines "
                    f"(retrying in {int(_FAIL_BACKOFF.total_seconds() // 60)} min)")


def _parse_rss(url: str, timeout: int = 6) -> list[str]:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ForexBot/2.0)"})
    with urlopen(req, timeout=timeout) as resp:
        content = resp.read()

    root = ET.fromstring(content)
    titles: list[str] = []

    # RSS 2.0
    for item in root.iter("item"):
        t = item.find("title")
        if t is not None and t.text and t.text.strip():
            titles.append(t.text.strip())

    # Atom 1.0
    ns = "{http://www.w3.org/2005/Atom}"
    for entry in root.iter(f"{ns}entry"):
        t = entry.find(f"{ns}title")
        if t is not None and t.text and t.text.strip():
            titles.append(t.text.strip())

    return titles[:30]


def _rank_for_symbol(headlines: list[str], symbol: str, max_n: int) -> list[str]:
    """Score headlines by keyword overlap with the traded symbol."""
    clean = symbol.upper().split(".")[0]  # strip .r / .pro broker suffixes
    keywords = _SYMBOL_KEYWORDS.get(clean, [])

    scored: list[tuple[int, str]] = []
    for h in headlines:
        h_lower = h.lower()
        score = sum(1 for kw in keywords if kw in h_lower)
        scored.append((score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [h for _, h in scored[:max_n]]

    if top:
        log.debug(f"NewsFeed top headline for {clean}: {top[0][:80]}")
    return top


def check_feeds() -> dict[str, str]:
    """Fetch every feed once and report how many headlines each returned."""
    report = {}
    for url in _RSS_FEEDS:
        try:
            n = len(_parse_rss(url))
            report[url] = f"OK — {n} headlines" if n else "EMPTY — reachable but no items"
        except Exception as e:
            report[url] = f"FAILED — {type(e).__name__}: {e}"
    return report


if __name__ == "__main__":
    for _url, _result in check_feeds().items():
        print(f"{_result:45s} {_url}")
