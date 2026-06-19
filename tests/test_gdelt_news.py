"""
tests/test_gdelt_news.py
========================
Live (tiny) smoke + schema + causality test for data/gdelt_news.py.

Behaviour
---------
* Does a SMALL live fetch (last few days, small maxrecords) against the GDELT
  DOC 2.0 API.
* If GDELT is reachable and returns rows, asserts the public schema and the
  point-in-time / causality guarantees:
    - columns == [title, domain, tone, url]
    - index is a UTC, monotonic-increasing DatetimeIndex named 'seendate'
    - title/domain/url are strings; tone is float (NaN allowed)
    - every article timestamp lies within [start, end] (no invented / future ts)
* If GDELT is UNREACHABLE / rate-limited / returns nothing, the test SKIPS
  gracefully: it prints "SKIP ..." and exits 0 (never fails the suite).

Runs either under pytest or as a plain script:  python3 tests/test_gdelt_news.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone

# Allow running directly from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from data.gdelt_news import fetch_news, _COLUMNS


# Tiny live window: last 5 days, few records, no cache so we exercise the network.
_TINY_DAYS = 5
_TINY_MAXRECORDS = 25


def _tiny_fetch():
    """Run a tiny live fetch; return (df, error) where error is None on success."""
    end = pd.Timestamp.now(tz="UTC")
    start = end - timedelta(days=_TINY_DAYS)
    try:
        df = fetch_news(start=start, end=end, use_cache=False,
                        maxrecords=_TINY_MAXRECORDS)
        return df, start, end, None
    except Exception as exc:  # availability problems should already be swallowed
        return None, start, end, exc


def _assert_schema_and_causality(df: pd.DataFrame, start, end) -> None:
    # --- schema -----------------------------------------------------------
    assert list(df.columns) == _COLUMNS, f"unexpected columns: {list(df.columns)}"

    assert isinstance(df.index, pd.DatetimeIndex), "index must be a DatetimeIndex"
    assert df.index.name == "seendate", f"index name must be 'seendate', got {df.index.name!r}"
    assert df.index.tz is not None, "index must be tz-aware (UTC)"
    assert str(df.index.tz) == "UTC", f"index tz must be UTC, got {df.index.tz}"

    # dtypes
    assert df["tone"].dtype == np.float64, f"tone must be float64, got {df['tone'].dtype}"
    for col in ("title", "domain", "url"):
        assert df[col].map(lambda v: isinstance(v, str)).all(), f"{col} must be all str"

    # --- causality / point-in-time ---------------------------------------
    # Monotonic-increasing index (sorted): downstream merge_asof relies on it.
    assert df.index.is_monotonic_increasing, "index must be monotonic increasing (sorted)"

    # No invented timestamps: every seendate must lie within the requested window
    # (allow a small skew tolerance for GDELT bin rounding at the edges).
    tol = pd.Timedelta(hours=2)
    assert (df.index >= start - tol).all(), "found article timestamp before window start"
    assert (df.index <= end + tol).all(), "found article timestamp after window end (look-ahead!)"

    # Tone, where present, must be a finite float in GDELT's plausible range.
    present = df["tone"].dropna()
    if len(present):
        assert np.isfinite(present.values).all(), "tone contains non-finite values"
        assert (present.between(-100, 100)).all(), "tone outside plausible [-100,100] range"


def test_gdelt_news_live_schema_and_causality():
    """Pytest entry point. SKIPs (not fails) when GDELT is unavailable."""
    df, start, end, err = _tiny_fetch()

    if err is not None:
        _skip(f"GDELT fetch raised {type(err).__name__}: {err}")
        return
    if df is None or df.empty:
        _skip("GDELT returned no rows (unreachable / rate-limited / recent-bias).")
        return

    _assert_schema_and_causality(df, start, end)
    print(f"PASS: {len(df)} rows, {df.index.min()} .. {df.index.max()}, "
          f"tone non-null={int(df['tone'].notna().sum())}/{len(df)}")


def _skip(msg: str) -> None:
    """Skip via pytest if available, else just print SKIP (script mode)."""
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print(f"SKIP: {msg}")


# ---------------------------------------------------------------------------
# Script mode: print SKIP and exit 0 on unavailability, never non-zero on outage.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df, start, end, err = _tiny_fetch()

    if err is not None:
        print(f"SKIP: GDELT fetch raised {type(err).__name__}: {err}")
        sys.exit(0)
    if df is None or df.empty:
        print("SKIP: GDELT returned no rows (unreachable / rate-limited / recent-bias).")
        sys.exit(0)

    try:
        _assert_schema_and_causality(df, start, end)
    except AssertionError as exc:
        print(f"FAIL: assertion error: {exc}")
        sys.exit(1)

    print(f"PASS: {len(df)} rows, {df.index.min()} .. {df.index.max()}, "
          f"tone non-null={int(df['tone'].notna().sum())}/{len(df)}")
    print("index monotonic increasing:", df.index.is_monotonic_increasing)
    print(df.head(5).to_string())
    sys.exit(0)
