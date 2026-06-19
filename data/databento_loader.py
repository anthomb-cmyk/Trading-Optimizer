"""
Databento CME Globex (GLBX.MDP3) historical OHLCV loader.

Fetches continuous-contract bars for ES, NQ, and GC (full-size CME front-month)
via the Databento Historical API and resamples to the requested timeframe.

Roll method (T3.2)
------------------
We use the **calendar roll** convention: symbol suffix `.c.0` with
`stype_in="continuous"`.  Calendar roll switches to the next contract on the
first business day of the expiry month (CME standard).  This gives a clean,
deterministic splice that matches what most institutional backtests assume.
Databento back-adjusts prices at each roll so there are no price gaps in the
returned series.

Volume-roll (`.v.0`) is available as an alternative but was not chosen here
because it introduces day-of-roll ambiguity that can create spurious signals.

Native GLBX.MDP3 OHLCV schemas
-------------------------------
Databento provides `ohlcv-1m`, `ohlcv-1h`, `ohlcv-1d` natively.
5m and 15m bars are built by downsampling from 1m bars in pandas.
That keeps the cost per credit proportional to 1m volume, not to the
number of timeframe variants requested.

Micro futures (MES, MNQ, MGC) vs full-size (ES, NQ, GC)
---------------------------------------------------------
Databento has limited continuous history for the micro contracts
(MES launched 2019, MNQ 2019, MGC 2010).  We map every micro symbol to
the same-underlying full-size continuous contract, which has history back
to the early 2000s on GLBX.MDP3.  Price and tick structure are identical;
only contract multiplier differs (handled outside the data layer in
config/settings.py INSTRUMENTS).

Cache
-----
Parquet files are written to DATA_DIR (config.settings.DATA_DIR).
Cache is keyed by (databento_symbol, timeframe) and respects
CACHE_EXPIRY_HOURS.  Re-fetches happen only when the cache is stale.
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import databento as db

from config.logger import get_logger
from config.settings import CACHE_EXPIRY_HOURS, DATA_DIR, LOOKBACK_DAYS
import config.run_logger as run_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------

# Map the short instrument keys used throughout this codebase to Databento
# continuous-contract symbols (calendar-roll, front month = .c.0).
# Micros map to the full-size contract for long-history continuous series.
_DB_SYMBOL_MAP: dict[str, str] = {
    # S&P 500 family
    "MES":   "ES.c.0",   # Micro E-mini S&P 500  -> full-size ES (calendar roll)
    "ES":    "ES.c.0",   # E-mini S&P 500
    "MES=F": "ES.c.0",
    "ES=F":  "ES.c.0",
    # Nasdaq-100 family
    "MNQ":   "NQ.c.0",   # Micro E-mini Nasdaq  -> full-size NQ
    "NQ":    "NQ.c.0",   # E-mini Nasdaq-100
    "MNQ=F": "NQ.c.0",
    "NQ=F":  "NQ.c.0",
    # Gold family
    "MGC":   "GC.c.0",   # Micro Gold Futures   -> full-size GC (COMEX/CME)
    "GC":    "GC.c.0",   # Standard Gold Futures
    "YG=F":  "GC.c.0",
    "MGC=F": "GC.c.0",
    "GC=F":  "GC.c.0",
}

# Databento native schemas (ohlcv-5m / ohlcv-15m do not exist on GLBX.MDP3).
# For 5m and 15m we fetch 1m bars and resample.
_NATIVE_SCHEMAS: dict[str, str] = {
    "1m":  "ohlcv-1m",
    "1h":  "ohlcv-1h",
    "4h":  "ohlcv-1h",   # fetched as 1h, resampled to 4h
    "1d":  "ohlcv-1d",
    "1w":  "ohlcv-1d",   # fetched as 1d, resampled to 1w
}
# Timeframes that need 1m base data for resampling
_RESAMPLE_FROM_1M: set[str] = {"5m", "15m", "3m"}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _db_cache_path(db_symbol: str, tf: str) -> Path:
    """Return parquet cache path for a given databento symbol + timeframe."""
    key = hashlib.md5(f"db_{db_symbol}_{tf}".encode()).hexdigest()[:8]
    safe = db_symbol.replace(".", "_").replace("=", "")
    return DATA_DIR / f"db_{safe}_{tf}_{key}.parquet"


def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age = time.time() - path.stat().st_mtime
    return age > CACHE_EXPIRY_HOURS * 3600


def _cache_covers(path: Path, requested_start: datetime) -> bool:
    """
    Return True if the parquet at *path* covers at least *requested_start*.

    Reads only the index from the parquet (cheap metadata read).  If the
    parquet's earliest bar is within a 1-day tolerance of *requested_start*
    the cache is considered sufficient; otherwise a refetch is needed.

    Returns False (needs refetch) if the file does not exist, cannot be read,
    or starts later than requested_start + 1 day.
    """
    if not path.exists():
        return False
    try:
        # Read only the index (avoids loading all columns)
        pf = pd.read_parquet(path, columns=[])
        if len(pf) == 0:
            return False
        cached_start = pf.index[0]
        # Normalise timezone: requested_start is UTC-aware
        if cached_start.tzinfo is None:
            cached_start = cached_start.tz_localize("UTC")
        else:
            cached_start = cached_start.tz_convert("UTC")
        if requested_start.tzinfo is None:
            requested_start = requested_start.replace(tzinfo=timezone.utc)
        # Allow 1-day slop (weekends, holidays near the boundary)
        tolerance = timedelta(days=1)
        return cached_start <= requested_start + tolerance
    except Exception as exc:
        log.warning("Could not read cache index for coverage check: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Integrity checks (T3.4)
# ---------------------------------------------------------------------------

def _is_cme_expected_break(t_prev: pd.Timestamp, t_next: pd.Timestamp) -> bool:
    """
    Return True if the gap between *t_prev* and *t_next* is an EXPECTED CME
    Globex equity-index break (ES / NQ / MES / MNQ) and should NOT be flagged.

    CME Globex session (UTC):
    - Trading runs Sunday ~23:00 UTC through Friday ~22:00 UTC.
    - Daily maintenance halt: ~22:00-23:00 UTC every weekday (17:00-18:00 ET).
    - Weekend closure: Friday ~22:00 UTC through Sunday ~23:00 UTC.

    A gap is "expected" if EITHER end falls outside active-session hours or
    crosses the Friday-close / Sunday-open boundary.
    """
    # Maintenance halt window: 21:55 – 23:05 UTC (5-min slop each side)
    HALT_START_UTC = 21 + 55 / 60   # 21:55 UTC
    HALT_END_UTC   = 23 + 5  / 60   # 23:05 UTC

    prev_dow   = t_prev.weekday()   # Mon=0 … Sun=6
    next_dow   = t_next.weekday()
    prev_hour  = t_prev.hour + t_prev.minute / 60.0
    next_hour  = t_next.hour + t_next.minute / 60.0

    # Weekend break: gap starts on Friday eve (≥21:55 UTC) or covers Sat/Sun
    # and ends on Sunday evening (≥22:55 UTC) or Monday morning
    if prev_dow == 4 and prev_hour >= HALT_START_UTC:  # Friday after halt start
        return True
    if prev_dow == 5 or prev_dow == 6:  # gap starts on Saturday or Sunday
        return True
    if next_dow == 6:  # gap ends on Sunday (before open)
        return True
    if next_dow == 0 and next_hour < HALT_END_UTC:  # early Sunday/Monday before open
        return True

    # Daily maintenance halt: gap crosses the 22:00-23:00 UTC window on any weekday
    if prev_hour >= HALT_START_UTC and next_hour <= HALT_END_UTC:
        return True
    # Gap that starts just before the halt and ends just after (same calendar day)
    if prev_hour >= HALT_START_UTC and next_hour <= HALT_END_UTC + 0.5:
        return True

    return False


def _check_integrity(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    run_uuid: Optional[str] = None,
) -> int:
    """
    Run integrity checks on a freshly fetched DataFrame.

    Checks performed:
    1. Monotonic increasing timestamps (no backward jumps).
    2. Unexpected weekend bars (Saturday/Sunday UTC).
    3. Gap detection: counts bars that are GENUINELY missing within an active
       CME Globex session.  Expected breaks (Friday close -> Sunday open,
       daily ~22:00-23:00 UTC maintenance halt) are NOT counted.
    4. Timezone correctness: index must be UTC-aware.

    Returns
    -------
    int
        Number of genuine integrity issues found (expected breaks excluded).
    """
    issues = 0

    # -- Timezone check
    if df.index.tz is None:
        run_logger.log_event(
            f"[{symbol} {timeframe}] Index has no timezone — expected UTC",
            run_uuid=run_uuid, level="WARN", kind="data",
        )
        issues += 1
    elif str(df.index.tz) != "UTC":
        run_logger.log_event(
            f"[{symbol} {timeframe}] Index timezone is {df.index.tz}, expected UTC",
            run_uuid=run_uuid, level="WARN", kind="data",
        )
        issues += 1

    # -- Monotonic check
    if not df.index.is_monotonic_increasing:
        run_logger.log_event(
            f"[{symbol} {timeframe}] Timestamps are not monotonic increasing",
            run_uuid=run_uuid, level="WARN", kind="data",
        )
        issues += 1

    # -- Weekend bars check
    # On CME Globex, Sunday bars CAN exist (session opens ~23:00 UTC Sunday).
    # True unexpected bars are on Saturday (dayofweek==5) or early Sunday
    # before the ~23:00 open. We only flag Saturday bars to avoid false positives.
    if len(df) > 0:
        saturday_mask = df.index.dayofweek == 5  # 5=Saturday
        n_saturday = int(saturday_mask.sum())
        if n_saturday > 0:
            run_logger.log_event(
                f"[{symbol} {timeframe}] {n_saturday} Saturday bars found (unexpected for CME Globex)",
                run_uuid=run_uuid, level="WARN", kind="data",
                payload={"n_saturday_bars": n_saturday},
            )
            issues += n_saturday

    # -- Gap detection for intraday timeframes (CME session-aware)
    _tf_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240,
    }
    if timeframe in _tf_minutes and len(df) > 1:
        bar_mins = _tf_minutes[timeframe]
        expected_delta = pd.Timedelta(minutes=bar_mins)
        gap_threshold = expected_delta * 2

        idx_series = df.index.to_series()
        diffs = idx_series.diff().dropna()
        raw_gaps = diffs[diffs > gap_threshold]

        real_gaps = 0
        expected_breaks = 0
        for t_next, delta in raw_gaps.items():
            # iloc position of t_next in the original index
            pos = df.index.get_loc(t_next)
            t_prev = df.index[pos - 1]
            if _is_cme_expected_break(t_prev, t_next):
                expected_breaks += 1
            else:
                real_gaps += 1

        if expected_breaks > 0:
            log.debug(
                "[%s %s] %d expected CME session breaks skipped "
                "(Friday close, daily halt 22:00-23:00 UTC)",
                symbol, timeframe, expected_breaks,
            )
        if real_gaps > 0:
            run_logger.log_event(
                f"[{symbol} {timeframe}] {real_gaps} REAL intraday gaps detected "
                f"(threshold: {gap_threshold}; {expected_breaks} expected CME breaks excluded)",
                run_uuid=run_uuid, level="WARN", kind="data",
                payload={"n_real_gaps": real_gaps, "n_expected_breaks": expected_breaks,
                         "gap_threshold_min": bar_mins * 2},
            )
            issues += real_gaps

    return issues


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def _get_dataset_end(
    client: db.Historical,
    dataset: str = "GLBX.MDP3",
    schema: str = "ohlcv-1m",
) -> datetime:
    """
    Query the Databento metadata API for the latest available timestamp in
    *dataset* for a given *schema*.  Returns a UTC datetime.

    The API returns per-schema end timestamps; we look up the schema-specific
    end first, then fall back to the top-level 'end' key, then to now-30m.
    """
    try:
        info = client.metadata.get_dataset_range(dataset=dataset)
        # Try schema-specific end first
        end_raw = None
        if isinstance(info, dict):
            schema_info = info.get("schema", {})
            if schema in schema_info:
                end_raw = schema_info[schema].get("end")
            if end_raw is None:
                end_raw = info.get("end")
        if end_raw is None:
            raise ValueError("No 'end' key in dataset range response")
        ts = pd.Timestamp(end_raw)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()
    except Exception as exc:
        log.warning("Could not determine dataset end time: %s -- falling back to now-30m", exc)
        return datetime.now(timezone.utc) - timedelta(minutes=30)


def _cap_end(end: datetime, dataset_end: datetime) -> datetime:
    """Return the earlier of *end* and *dataset_end*."""
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if dataset_end.tzinfo is None:
        dataset_end = dataset_end.replace(tzinfo=timezone.utc)
    return min(end, dataset_end)


def _fetch_databento_1m(
    db_symbol: str,
    start: datetime,
    end: datetime,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV bars from Databento GLBX.MDP3 for *db_symbol*.

    Parameters
    ----------
    db_symbol : str
        Databento continuous-contract symbol, e.g. ``"ES.c.0"``.
    start : datetime
        Inclusive start (UTC).
    end : datetime
        Exclusive end (UTC).
    api_key : str, optional
        Databento API key.  Falls back to ``DATABENTO_API_KEY`` env var.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC), lowercase columns: open, high, low, close, volume.
    """
    key = api_key or os.getenv("DATABENTO_API_KEY")
    if not key:
        raise EnvironmentError(
            "DATABENTO_API_KEY not set.  Add it to .env or the environment."
        )

    client = db.Historical(key)
    # Cap end to dataset availability to avoid 422 errors during live/near-live hours
    dataset_end = _get_dataset_end(client, schema="ohlcv-1m")
    end = _cap_end(end, dataset_end)
    log.info(
        "Databento fetch: %s ohlcv-1m  %s -> %s",
        db_symbol,
        start.strftime("%Y-%m-%dT%H:%M"),
        end.strftime("%Y-%m-%dT%H:%M"),
    )
    store = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=db_symbol,
        schema="ohlcv-1m",
        stype_in="continuous",
        start=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end=end.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    raw_df = store.to_df()
    return _normalize_df(raw_df)


def _fetch_databento_native(
    db_symbol: str,
    schema: str,
    start: datetime,
    end: datetime,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a natively supported OHLCV schema (1m, 1h, 1d)."""
    key = api_key or os.getenv("DATABENTO_API_KEY")
    if not key:
        raise EnvironmentError("DATABENTO_API_KEY not set.")

    client = db.Historical(key)
    # Cap end to dataset availability to avoid 422 errors during live/near-live hours
    dataset_end = _get_dataset_end(client, schema=schema)
    end = _cap_end(end, dataset_end)
    log.info(
        "Databento fetch: %s %s  %s -> %s",
        db_symbol, schema,
        start.strftime("%Y-%m-%dT%H:%M"),
        end.strftime("%Y-%m-%dT%H:%M"),
    )
    store = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=db_symbol,
        schema=schema,
        stype_in="continuous",
        start=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end=end.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    raw_df = store.to_df()
    return _normalize_df(raw_df)


def _normalize_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a Databento to_df() result to the standard loader format:
    - UTC DatetimeIndex named 'datetime'
    - lowercase columns: open, high, low, close, volume
    - prices divided by FIXED_PRICE_SCALE (1e9) if still in integer form
    """
    df = raw_df.copy()

    # Databento returns prices as fixed-point integers (×1e9) in some schemas.
    # to_df(price_type='float') already converts, but the default call may not.
    # Check if prices look like raw integers (> 1e6 for typical futures prices).
    for col in ("open", "high", "low", "close"):
        if col in df.columns and df[col].dtype in (np.int64, np.float64):
            if df[col].abs().max() > 1e6 * 1e3:  # > 1e9 means still raw fixed-point
                df[col] = df[col] / db.FIXED_PRICE_SCALE

    # Ensure lowercase column names
    df.columns = [c.lower() for c in df.columns]

    # Keep only OHLCV; drop Databento-specific columns (rtype, publisher_id, etc.)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]

    # Make sure index is UTC DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "datetime"
    return df.dropna(subset=["open", "high", "low", "close"])


def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample a DataFrame of OHLCV bars to a coarser timeframe.

    Uses the standard OHLCV aggregation (first open, max high, min low,
    last close, sum volume).  Drops empty bars (no trades in that period).
    """
    _rule_map = {
        "3m":  "3min",
        "5m":  "5min",
        "15m": "15min",
        "4h":  "4h",
        "1w":  "W-MON",  # week starting Monday
    }
    rule = _rule_map.get(timeframe)
    if rule is None:
        raise ValueError(f"Cannot resample to unsupported timeframe '{timeframe}'")

    resampled = df.resample(rule, label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    resampled.index.name = "datetime"
    return resampled


# ---------------------------------------------------------------------------
# Public API (T3.1)
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    days: int = LOOKBACK_DAYS,
    use_cache: bool = True,
    api_key: Optional[str] = None,
    run_uuid: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars for *symbol* at *timeframe* from Databento GLBX.MDP3.

    Supported timeframes: 1m, 3m, 5m, 15m, 1h, 4h, 1d, 1w.
    Supported symbols: MES, ES, MNQ, NQ, MGC, GC (and their =F variants).

    Parameters
    ----------
    symbol : str
        Instrument key as used in the rest of the codebase (e.g. ``"MES"``).
        Micros (MES, MNQ, MGC) are mapped to the full-size continuous contract.
    timeframe : str
        Bar period string: ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"4h"``,
        ``"1d"``, ``"1w"``.
    days : int
        Number of calendar days of history to fetch.
    use_cache : bool
        If True (default) return cached parquet when it is still fresh
        (within CACHE_EXPIRY_HOURS).  Set to False to force a fresh fetch.
    api_key : str, optional
        Databento API key.  Falls back to ``DATABENTO_API_KEY`` env var.
    run_uuid : str, optional
        Run UUID for run_logger.  Pass the result of ``run_logger.start_run()``
        to attach data-fetch events to a run.

    Returns
    -------
    pd.DataFrame
        UTC DatetimeIndex named ``'datetime'``, columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``.
        Identical shape to what ``_add_indicators`` expects.

    Notes
    -----
    Roll method: calendar roll (.c.0), see module docstring.
    5m and 15m bars are resampled from native 1m data.
    4h bars are resampled from native 1h data.
    """
    # Resolve to Databento continuous-contract symbol
    db_symbol = _DB_SYMBOL_MAP.get(symbol)
    if db_symbol is None:
        raise ValueError(
            f"Symbol '{symbol}' is not mapped to a Databento continuous contract. "
            f"Known symbols: {sorted(_DB_SYMBOL_MAP)}"
        )

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    cache_path = _db_cache_path(db_symbol, timeframe)
    if use_cache and not _is_stale(cache_path) and _cache_covers(cache_path, start_dt):
        log.debug(
            "Loading %s %s from Databento cache: %s (covers %s)",
            symbol, timeframe, cache_path.name, start_dt.strftime("%Y-%m-%d"),
        )
        df = pd.read_parquet(cache_path)
        # Ensure UTC index (parquet may strip tz)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        # Slice to the requested window (don't return extra history)
        mask = df.index >= pd.Timestamp(start_dt)
        return df[mask]
    elif use_cache and cache_path.exists() and not _cache_covers(cache_path, start_dt):
        log.info(
            "Cache for %s %s does not cover requested start %s — refetching %d days",
            symbol, timeframe, start_dt.strftime("%Y-%m-%d"), days,
        )

    # -- Decide fetch strategy
    if timeframe in _RESAMPLE_FROM_1M:
        # Fetch 1m, then resample
        raw = _fetch_databento_1m(db_symbol, start_dt, end_dt, api_key)
        df = _resample(raw, timeframe)
        native_schema = "ohlcv-1m"
    elif timeframe in _NATIVE_SCHEMAS:
        native_schema = _NATIVE_SCHEMAS[timeframe]
        df_native = _fetch_databento_native(db_symbol, native_schema, start_dt, end_dt, api_key)
        if timeframe in ("4h", "1w"):
            df = _resample(df_native, timeframe)
        else:
            df = df_native
    else:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            "Supported: 1m, 3m, 5m, 15m, 1h, 4h, 1d, 1w"
        )

    # -- Integrity checks (T3.4)
    n_issues = _check_integrity(df, symbol, timeframe, run_uuid=run_uuid)
    ok = (n_issues == 0)

    # -- Log fetch to run_logger (T3.4)
    bar_start = str(df.index[0]) if len(df) > 0 else None
    bar_end   = str(df.index[-1]) if len(df) > 0 else None
    run_logger.log_data_fetch(
        source="databento",
        symbol=symbol,
        run_uuid=run_uuid,
        timeframe=timeframe,
        bar_start=bar_start,
        bar_end=bar_end,
        n_bars=len(df),
        gaps_found=n_issues,
        ok=ok,
    )

    # -- Cache
    df.to_parquet(cache_path, engine="pyarrow", compression="snappy")
    log.info(
        "Cached Databento %s %s -> %s (%d bars, %d integrity issues)",
        symbol, timeframe, cache_path.name, len(df), n_issues,
    )

    return df
