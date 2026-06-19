"""
data/gdelt_news.py
==================
CAUSAL, point-in-time GDELT macro-news ingestion for the APEX Optimizer.

Purpose
-------
Pull a stream of macro / market news headlines (S&P 500 / MES context) from the
GDELT DOC 2.0 API, tagged with each article's *publish* timestamp (``seendate``)
and GDELT's built-in **tone** (negative = bearish, positive = bullish).  The
output is a tidy ``pandas.DataFrame`` that downstream strategy / regime code can
join against price bars under a strict no-look-ahead rule.

Causality contract
------------------
Every row carries the article's GDELT ``seendate`` (UTC) as its index.  The
timestamp is *never invented*: it is taken verbatim from the GDELT record and
parsed as UTC.  Downstream code that wants to use news as a feature at bar time
``t`` MUST filter ``news.loc[news.index <= t]`` (point-in-time).  This module
does not shift, forward-fill, or otherwise smear timestamps, so the only
look-ahead risk is in the consumer, not here.

Tone is sourced from GDELT's own ``timelinetone`` series for the *same query and
window* and mapped onto each article by the time bin its ``seendate`` falls in.
GDELT computes that average tone from documents *as they were published*, so it
introduces no future information relative to the article's own timestamp.  If the
tone series is unavailable (rate-limited / unreachable) tone degrades to ``NaN``
rather than being fabricated.

GDELT DOC 2.0 history limits  (READ THIS before backtesting on it)
-----------------------------------------------------------------
* The DOC 2.0 ``artlist`` index favours the **most recent ~3 months** of
  coverage and is strongly biased toward recent material.  Requests for deep
  multi-year history will return *partial* results or nothing for old windows.
* For reliable multi-year macro-news history you must use the GDELT **GKG**
  (Global Knowledge Graph) files or the GDELT **BigQuery** ``gdelt-bq`` dataset,
  which is out of scope for this lightweight live module.
* GDELT rate-limits the public API to roughly **one request every 5 seconds**;
  bursts return HTTP 429.  This module spaces its calls and backs off on 429.

Because of the above, ``fetch_news`` is written to **degrade gracefully**: on any
network error, 429, empty window, or schema surprise it logs the coverage it
actually obtained and returns whatever (possibly empty) well-typed DataFrame it
could build — it never raises for an upstream/availability problem.

Schema (return value of :func:`fetch_news`)
-------------------------------------------
``pandas.DataFrame`` indexed by ``seendate`` (``DatetimeIndex``, tz=UTC, sorted
ascending), columns:

    title   : str    article headline
    domain  : str    publisher domain (e.g. "finance.yahoo.com")
    tone    : float  GDELT tone for the article's time bin (NaN if unavailable)
    url     : str    canonical article URL

Cache
-----
Results are cached to parquet under ``config.settings.DATA_DIR``, keyed by a hash
of ``(query, start, end)``.  Cache hits respect ``CACHE_EXPIRY_HOURS``.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:  # the repo's colourised logger; fall back to stdlib so the module is import-safe
    from config.logger import get_logger
    log = get_logger(__name__)
except Exception:  # pragma: no cover - defensive only
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

try:
    from config.settings import DATA_DIR, CACHE_EXPIRY_HOURS
except Exception:  # pragma: no cover - allow standalone use
    DATA_DIR = Path(__file__).resolve().parent / "cache"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_EXPIRY_HOURS = 4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Default macro / market query tuned for S&P 500 / MES context.  English only to
# keep tone comparable.  GDELT query syntax: quoted phrases = exact, OR for union.
DEFAULT_QUERY = (
    '("stock market" OR "Federal Reserve" OR "S&P 500" OR "interest rates" '
    'OR inflation OR recession OR economy) sourcelang:eng'
)

# GDELT public-API politeness: ~1 request / 5s.  We use a small margin.
_MIN_REQUEST_INTERVAL_S = 5.5
_MAX_RETRIES = 3
_HEADERS = {"User-Agent": "APEX-Optimizer/1.0 (macro-news research; requests)"}

# DOC 2.0 artlist caps maxrecords at 250.
_ARTLIST_MAX = 250

# Final, stable output schema.
_COLUMNS = ["title", "domain", "tone", "url"]

# Module-level throttle clock so repeated calls in one process stay polite.
_last_request_ts: float = 0.0


# ---------------------------------------------------------------------------
# Low-level HTTP helper (polite + 429-aware)
# ---------------------------------------------------------------------------
def _throttled_get(params: dict, timeout: int = 30) -> Optional[requests.Response]:
    """GET the GDELT DOC endpoint, respecting the rate limit and retrying on 429.

    Returns the ``Response`` on HTTP 200, or ``None`` if the request could not be
    completed successfully (network error, persistent 429, non-200).  Never
    raises for availability problems — the caller degrades gracefully.
    """
    global _last_request_ts
    for attempt in range(1, _MAX_RETRIES + 1):
        wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(GDELT_DOC_URL, params=params, headers=_HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            log.warning("GDELT request error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
            _last_request_ts = time.monotonic()
            continue
        _last_request_ts = time.monotonic()

        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            backoff = _MIN_REQUEST_INTERVAL_S * attempt
            log.warning("GDELT rate-limited (429); backing off %.1fs (attempt %d/%d)",
                        backoff, attempt, _MAX_RETRIES)
            time.sleep(backoff)
            continue
        log.warning("GDELT returned HTTP %d (attempt %d/%d): %s",
                    resp.status_code, attempt, _MAX_RETRIES, resp.text[:200])
    return None


# ---------------------------------------------------------------------------
# Date / cache helpers
# ---------------------------------------------------------------------------
def _to_utc(ts) -> Optional[pd.Timestamp]:
    """Coerce a date-ish value to a tz-aware UTC ``pd.Timestamp`` (or None)."""
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _gdelt_dt(ts: pd.Timestamp) -> str:
    """Format a UTC timestamp as GDELT's ``YYYYMMDDHHMMSS`` datetime string."""
    return ts.tz_convert("UTC").strftime("%Y%m%d%H%M%S")


def _cache_path(query: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Path:
    """Parquet cache path keyed by a hash of (query, start, end)."""
    s = start.isoformat() if start is not None else "auto"
    e = end.isoformat() if end is not None else "auto"
    key = hashlib.md5(f"{query}|{s}|{e}".encode()).hexdigest()[:10]
    return Path(DATA_DIR) / f"gdelt_news_{key}.parquet"


def _cache_fresh(path: Path) -> bool:
    """True if *path* exists and is younger than CACHE_EXPIRY_HOURS."""
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h < CACHE_EXPIRY_HOURS


def _empty_frame() -> pd.DataFrame:
    """An empty, correctly-typed/indexed DataFrame matching the public schema."""
    idx = pd.DatetimeIndex([], tz="UTC", name="seendate")
    return pd.DataFrame(
        {
            "title": pd.Series([], dtype="object"),
            "domain": pd.Series([], dtype="object"),
            "tone": pd.Series([], dtype="float64"),
            "url": pd.Series([], dtype="object"),
        },
        index=idx,
    )


def _parse_seendate(value: str) -> Optional[pd.Timestamp]:
    """Parse GDELT seendate ('YYYYMMDDTHHMMSSZ') into a UTC Timestamp."""
    if not value:
        return None
    try:
        t = pd.to_datetime(value, format="%Y%m%dT%H%M%SZ", utc=True)
    except (ValueError, TypeError):
        try:
            t = pd.to_datetime(value, utc=True)
        except (ValueError, TypeError):
            return None
    return t


# ---------------------------------------------------------------------------
# GDELT fetch pieces
# ---------------------------------------------------------------------------
def _fetch_articles(query: str, start: pd.Timestamp, end: pd.Timestamp,
                    maxrecords: int) -> pd.DataFrame:
    """Fetch the article list (no tone) for the window.  Returns indexed frame."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": int(min(max(maxrecords, 1), _ARTLIST_MAX)),
        "startdatetime": _gdelt_dt(start),
        "enddatetime": _gdelt_dt(end),
        "sort": "datedesc",
    }
    resp = _throttled_get(params)
    if resp is None:
        log.warning("GDELT artlist unavailable for window %s .. %s", start, end)
        return _empty_frame()

    try:
        payload = resp.json()
    except ValueError:
        # GDELT sometimes returns a plain-text error with HTTP 200.
        log.warning("GDELT artlist returned non-JSON body: %s", resp.text[:200])
        return _empty_frame()

    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    if not articles:
        log.info("GDELT artlist returned 0 articles for window %s .. %s", start, end)
        return _empty_frame()

    rows = []
    for art in articles:
        ts = _parse_seendate(art.get("seendate", ""))
        if ts is None:
            continue  # never invent a timestamp; drop rows we cannot place in time
        rows.append(
            {
                "seendate": ts,
                "title": str(art.get("title", "") or ""),
                "domain": str(art.get("domain", "") or ""),
                "url": str(art.get("url", "") or ""),
            }
        )

    if not rows:
        return _empty_frame()

    df = pd.DataFrame(rows).set_index("seendate")
    df.index.name = "seendate"
    return df


def _fetch_tone_timeline(query: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    """Fetch GDELT's average-tone time series (timelinetone) for the window.

    Returns a UTC-indexed Series of tone values, or ``None`` if unavailable.
    This is GDELT's own per-bin average tone computed from documents as published;
    mapping an article onto its time bin introduces no future information.
    """
    params = {
        "query": query,
        "mode": "timelinetone",
        "format": "json",
        "startdatetime": _gdelt_dt(start),
        "enddatetime": _gdelt_dt(end),
    }
    resp = _throttled_get(params)
    if resp is None:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None

    timeline = payload.get("timeline", []) if isinstance(payload, dict) else []
    if not timeline:
        return None
    # timeline is a list of series; the tone series is the first (and only) one.
    series = timeline[0].get("data", [])
    pts = []
    for p in series:
        ts = _parse_seendate(p.get("date", ""))
        if ts is None:
            continue
        try:
            val = float(p.get("value"))
        except (TypeError, ValueError):
            continue
        pts.append((ts, val))
    if not pts:
        return None
    s = pd.Series({t: v for t, v in pts}).sort_index()
    s.index = pd.DatetimeIndex(s.index, tz="UTC")
    return s


def _attach_tone(articles: pd.DataFrame, tone: Optional[pd.Series]) -> pd.DataFrame:
    """Map each article's tone from the nearest *prior* timeline bin (causal)."""
    if articles.empty:
        articles["tone"] = pd.Series([], dtype="float64")
        return articles
    if tone is None or tone.empty:
        articles["tone"] = np.nan
        return articles

    art_sorted = articles.sort_index()
    tone_sorted = tone.sort_index()
    # merge_asof with direction="backward": each article gets the tone of the
    # most recent timeline bin at or before its seendate -> strictly causal.
    merged = pd.merge_asof(
        art_sorted.reset_index(),
        tone_sorted.rename("tone").reset_index().rename(columns={"index": "seendate"}),
        on="seendate",
        direction="backward",
    )
    merged = merged.set_index("seendate")
    merged.index.name = "seendate"
    return merged


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the public schema: columns, dtypes, sorted UTC index."""
    out = df.copy()
    for col in _COLUMNS:
        if col not in out.columns:
            out[col] = np.nan if col == "tone" else ""
    out = out[_COLUMNS]
    out["title"] = out["title"].astype("object")
    out["domain"] = out["domain"].astype("object")
    out["url"] = out["url"].astype("object")
    out["tone"] = pd.to_numeric(out["tone"], errors="coerce").astype("float64")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "seendate"
    out = out.sort_index()
    # Drop exact duplicate (timestamp, url) pairs GDELT occasionally repeats.
    out = out[~out.reset_index().duplicated(subset=["seendate", "url"]).values]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_news(
    query: Optional[str] = None,
    start=None,
    end=None,
    use_cache: bool = True,
    maxrecords: int = _ARTLIST_MAX,
) -> pd.DataFrame:
    """Fetch point-in-time GDELT macro-news as a tidy DataFrame.

    Parameters
    ----------
    query : str, optional
        GDELT DOC 2.0 query string.  Defaults to :data:`DEFAULT_QUERY`
        (S&P 500 / Federal Reserve / inflation / recession / economy, English).
    start, end : date-like, optional
        Inclusive UTC window.  Naive inputs are assumed UTC.  Defaults to the
        last 7 days (``end = now``).  Note GDELT favours recent months; very old
        windows return partial/empty results (see module docstring).
    use_cache : bool, default True
        Use a fresh parquet cache hit (< CACHE_EXPIRY_HOURS) if present.
    maxrecords : int, default 250
        Max articles to request (GDELT caps artlist at 250).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``seendate`` (UTC, sorted), columns ``[title, domain, tone, url]``.
        Empty (but correctly typed) if GDELT is unreachable or the window is bare.
        Never raises for availability problems.
    """
    query = query or DEFAULT_QUERY
    end_ts = _to_utc(end) or pd.Timestamp.now(tz="UTC")
    start_ts = _to_utc(start) or (end_ts - timedelta(days=7))
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts

    cache_path = _cache_path(query, start_ts, end_ts)
    if use_cache and _cache_fresh(cache_path):
        try:
            cached = pd.read_parquet(cache_path)
            cached = _finalize(cached)
            log.info("GDELT news cache hit: %s (%d rows)", cache_path.name, len(cached))
            return cached
        except Exception as exc:  # corrupt cache -> fall through to live fetch
            log.warning("GDELT news cache unreadable (%s); refetching", exc)

    log.info("Fetching GDELT news | %s .. %s | maxrecords=%d", start_ts, end_ts, maxrecords)
    articles = _fetch_articles(query, start_ts, end_ts, maxrecords)

    tone = None
    if not articles.empty:
        tone = _fetch_tone_timeline(query, start_ts, end_ts)
        if tone is None:
            log.warning("GDELT tone timeline unavailable; tone will be NaN for this window")

    df = _attach_tone(articles, tone)
    df = _finalize(df)

    # Coverage report (the graceful-degradation contract).
    if df.empty:
        log.warning("GDELT coverage: 0 articles for %s .. %s (recent-bias / rate-limit / outage)",
                    start_ts.date(), end_ts.date())
    else:
        tone_cov = float(df["tone"].notna().mean()) * 100.0
        log.info("GDELT coverage: %d articles, %s .. %s, tone populated %.0f%%",
                 len(df), df.index.min(), df.index.max(), tone_cov)

    # Cache even empty frames so repeated misses don't hammer the rate limit;
    # an empty cache simply expires after CACHE_EXPIRY_HOURS.
    if use_cache:
        try:
            df.to_parquet(cache_path, engine="pyarrow", compression="snappy")
        except Exception as exc:  # pragma: no cover - non-fatal
            log.warning("Could not write GDELT news cache (%s): %s", cache_path.name, exc)

    return df


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    out = fetch_news()
    print(out.head(10))
    print(f"\n{len(out)} rows | tone non-null: {out['tone'].notna().sum()}")
    if not out.empty:
        print("index monotonic increasing:", out.index.is_monotonic_increasing)
        print("index tz:", out.index.tz)
