"""
News event filter — blackout window around high-impact economic events.

Data source: FairEconomy / Forex Factory calendar API (current week only).
Cached to disk and refreshed once per day.

Blackout logic:
    New entries are blocked from (event_time - before_mins) to
    (event_time + after_mins).  Existing positions are NOT affected —
    this filter only gates new signal entries.

For historical backtests the API returns only the current week, so the
blackout mask will be all-False for dates outside that window. That is
intentional: a historical events feed can be plugged in later.
"""
import json
import time
from datetime import timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from config.logger import get_logger
from config.settings import DATA_DIR, NEWS

log = get_logger(__name__)

_CACHE_FILE  = DATA_DIR / "news_events_cache.json"
_CACHE_MAX_S = 24 * 3600   # re-fetch at most once per day


class NewsFilter:
    """
    Fetches the economic calendar, caches it, and produces a blackout mask
    for any DataFrame index.
    """

    def __init__(
        self,
        api_url:              str  = NEWS["api_url"],
        high_impact_only:     bool = NEWS["high_impact_only"],
        blackout_before_mins: int  = NEWS["blackout_before_mins"],
        blackout_after_mins:  int  = NEWS["blackout_after_mins"],
        currencies:           list = None,
        cache_file:           Path = _CACHE_FILE,
    ):
        self.api_url              = api_url
        self.high_impact_only     = high_impact_only
        self.blackout_before_secs = blackout_before_mins * 60
        self.blackout_after_secs  = blackout_after_mins  * 60
        self.currencies           = currencies if currencies is not None else NEWS["currencies"]
        self.cache_file           = cache_file
        self._events: List[dict]  = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_events(self, force_refresh: bool = False) -> None:
        """Fetch (or restore from cache) the economic calendar."""
        if not force_refresh and self._is_cache_fresh():
            self._events = self._read_cache()
            log.debug("Loaded %d news events from cache.", len(self._events))
            return

        try:
            self._events = self._fetch_remote()
            self._write_cache(self._events)
            log.info("Fetched %d news events from API.", len(self._events))
        except Exception as exc:
            log.warning("News API fetch failed (%s) — falling back to cache.", exc)
            if self.cache_file.exists():
                self._events = self._read_cache()
                log.info("Using cached news events (%d events).", len(self._events))
            else:
                log.warning("No news cache available — filter will allow all bars.")
                self._events = []

    def get_blackout_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Boolean Series aligned to df.index.
        True  = this bar is inside a blackout window → block new entries.
        False = safe to trade.
        """
        if not self._events:
            return pd.Series(False, index=df.index)

        # Convert index to integer seconds (UTC) once for vectorised comparison
        bar_ts = df.index.astype("int64") // 1_000_000_000

        blocked = pd.array([False] * len(df))
        for ev in self._events:
            lo = ev["timestamp"] - self.blackout_before_secs
            hi = ev["timestamp"] + self.blackout_after_secs
            blocked |= (bar_ts >= lo) & (bar_ts <= hi)

        return pd.Series(blocked, index=df.index, dtype=bool)

    def is_blackout(self, dt) -> bool:
        """Single-datetime check for live trading."""
        if not self._events:
            return False
        if getattr(dt, "tzinfo", None) is None:
            ts = dt.replace(tzinfo=timezone.utc).timestamp()
        else:
            ts = dt.timestamp()
        for ev in self._events:
            if (ev["timestamp"] - self.blackout_before_secs) <= ts <= (ev["timestamp"] + self.blackout_after_secs):
                return True
        return False

    # ── Remote fetch ───────────────────────────────────────────────────────────

    def _fetch_remote(self) -> List[dict]:
        resp = requests.get(self.api_url, timeout=15, verify=False)
        resp.raise_for_status()
        return self._parse_events(resp.json())

    def _parse_events(self, raw: list) -> List[dict]:
        """
        Parse the FairEconomy / Forex Factory JSON format.
        Expected fields: country, impact, date, time, title.
        Time is assumed to be US/Eastern (ET) when no tz info is present.
        """
        import pytz
        eastern = pytz.timezone("America/New_York")
        events  = []

        for item in raw:
            currency = item.get("country") or item.get("currency", "")
            if self.currencies and currency not in self.currencies:
                continue

            impact = (item.get("impact") or "").lower()
            if self.high_impact_only and impact not in ("high", "red", "3"):
                continue

            date_str = item.get("date", "")
            time_str = item.get("time", "00:00:00") or "00:00:00"
            if not date_str:
                continue

            try:
                dt = pd.to_datetime(f"{date_str} {time_str}")
                if dt.tzinfo is None:
                    dt = eastern.localize(dt.to_pydatetime()).astimezone(pytz.utc)
                else:
                    dt = dt.tz_convert("UTC").to_pydatetime()
                events.append({
                    "timestamp": int(dt.timestamp()),
                    "title":     item.get("title", ""),
                })
            except Exception:
                continue

        return events

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _is_cache_fresh(self) -> bool:
        if not self.cache_file.exists():
            return False
        return (time.time() - self.cache_file.stat().st_mtime) < _CACHE_MAX_S

    def _read_cache(self) -> List[dict]:
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_cache(self, events: List[dict]) -> None:
        try:
            self.cache_file.write_text(json.dumps(events, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("Could not write news cache: %s", exc)
