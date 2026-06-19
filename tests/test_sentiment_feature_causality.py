"""
tests/test_sentiment_feature_causality.py
==========================================
Causality regression tests for strategies/sentiment_feature.py.

All tests use SYNTHETIC in-memory DataFrames only.
NO live GDELT network calls are made.

Three guarantees verified
--------------------------
1. Point-in-time: sent_score / sent_z at bar t never incorporate news
   with timestamp > t (no look-ahead).

2. Append-invariance (no-repaint): computing features on df[:k] vs
   df[:k+50] yields identical sent_score / sent_z values for settled
   indices well within the shorter prefix.

3. Empty-news safety: passing an empty news DataFrame fills the three
   columns with zeros/NaN without raising.

Supports:  python3 tests/test_sentiment_feature_causality.py
           pytest tests/test_sentiment_feature_causality.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# Allow running directly from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.sentiment_feature import add_sentiment_features


# ── Synthetic data factories ──────────────────────────────────────────────────

def _make_price_df(n: int = 200, freq: str = "15min") -> pd.DataFrame:
    """Reproducible synthetic price bars with a UTC DatetimeIndex."""
    rng = np.random.default_rng(42)
    close = 4000.0 + np.cumsum(rng.standard_normal(n) * 2.0)
    atr   = np.abs(rng.standard_normal(n)) * 5.0 + 3.0
    idx   = pd.date_range("2024-01-01 00:00", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open":  close - rng.uniform(0, 2, n),
            "high":  close + rng.uniform(0.5, 3, n),
            "low":   close - rng.uniform(0.5, 3, n),
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
            "atr":   atr,
        },
        index=idx,
    )


def _make_news_df(
    n: int = 80,
    start: str = "2024-01-01 00:30",
    freq: str = "30min",
    seed: int = 7,
) -> pd.DataFrame:
    """
    Reproducible synthetic news DataFrame with UTC DatetimeIndex (seendate).
    Tone values span [-5, +5] to mimic GDELT's range.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "title":  [f"Headline {i}" for i in range(n)],
            "domain": ["example.com"] * n,
            "tone":   rng.uniform(-5.0, 5.0, n),
            "url":    [f"http://example.com/{i}" for i in range(n)],
        },
        index=idx,
    )


def _empty_news_df() -> pd.DataFrame:
    """Correctly-typed empty news DataFrame matching the gdelt_news schema."""
    idx = pd.DatetimeIndex([], tz="UTC", name="seendate")
    return pd.DataFrame(
        {
            "title":  pd.Series([], dtype="object"),
            "domain": pd.Series([], dtype="object"),
            "tone":   pd.Series([], dtype="float64"),
            "url":    pd.Series([], dtype="object"),
        },
        index=idx,
    )


# ── Test 1: point-in-time / no look-ahead ────────────────────────────────────

def test_no_lookahead():
    """
    For every bar t, sent_score at t must not change when future news
    (timestamps > bar[t]) are appended to the news DataFrame.

    Method: compute features on the full price_df with only causal news
    (ts <= bar_ts), then append future news and confirm sent_score[t] is
    unchanged.

    We test this by computing on a prefix of the NEWS frame that stops
    at bar_ts and comparing to the full-news result at that bar.
    The single-bar-frame approach is NOT used because the empty-news fast-path
    would mask real look-ahead issues; instead we always pass the full price
    frame and only restrict the news.
    """
    price_df = _make_price_df(n=100)
    news_df  = _make_news_df(n=60)

    # Compute with all news — this is the "future" reference
    result_full = add_sentiment_features(price_df, news_df,
                                         lookback_hours=6, halflife_hours=3)

    # Re-compute for each bar with only causally-visible news
    # To avoid the empty-news fast-path (which returns 0.0) skewing comparisons,
    # we run on the FULL price frame but restrict news to ts <= bar_ts.
    # The key property: sent_score at position i must equal the value produced
    # by restricting news to ts <= price_df.index[i].
    violations = []
    for i, bar_ts in enumerate(price_df.index):
        causal_news = news_df[news_df.index <= bar_ts]

        # Run full price frame with causal-only news
        result_causal_run = add_sentiment_features(
            price_df,
            causal_news,
            lookback_hours=6,
            halflife_hours=3,
        )

        full_val   = result_full["sent_score"].iloc[i]
        causal_val = result_causal_run["sent_score"].iloc[i]

        # Both NaN -> OK (no news in window either way)
        if (np.isnan(full_val) if isinstance(full_val, float) else False) and \
           (np.isnan(causal_val) if isinstance(causal_val, float) else False):
            continue

        # Zero vs NaN: when no news in window, full run gives NaN; causal
        # restricted-news run may also give NaN. Both are acceptable as "no data".
        full_is_empty   = (np.isnan(full_val)   if isinstance(full_val, float)   else False) or full_val == 0.0
        causal_is_empty = (np.isnan(causal_val) if isinstance(causal_val, float) else False) or causal_val == 0.0
        if full_is_empty and causal_is_empty:
            continue

        # Both finite and non-trivial -> must match
        if not (np.isnan(full_val) if isinstance(full_val, float) else False) and \
           not (np.isnan(causal_val) if isinstance(causal_val, float) else False):
            if not np.isclose(float(full_val), float(causal_val), rtol=1e-9, atol=1e-12):
                violations.append((i, bar_ts, full_val, causal_val))

    assert not violations, (
        f"Look-ahead detected at {len(violations)} bar(s). "
        f"First: bar={violations[0][1]}, full={violations[0][2]:.6f}, "
        f"causal={violations[0][3]:.6f}"
    )
    print(f"[PASS] test_no_lookahead — {len(price_df)} bars checked, "
          f"no look-ahead detected")


# ── Test 2: append-invariance (no-repaint) ────────────────────────────────────

def test_append_invariance():
    """
    Computing features on df[:k] must produce identical sent_score and
    sent_z for settled bars as computing on df[:k+50].

    'Settled' means bar index < k - z_roll_window (z-score window=20 bars).
    We also conservatively require the same news subset (causal for that bar),
    so we split news by bar[k].ts as well.
    """
    price_df = _make_price_df(n=300)
    news_df  = _make_news_df(n=120)

    k         = 200
    z_window  = 20     # rolling window used in add_sentiment_features
    settled   = k - z_window - 5   # conservative settled zone

    assert settled > 30, "Need more settled bars for a meaningful test"

    cutoff_ts = price_df.index[k - 1]

    # Short prefix: first k bars, news up to bar[k-1]
    price_short = price_df.iloc[:k]
    news_short  = news_df[news_df.index <= cutoff_ts]
    res_short   = add_sentiment_features(price_short, news_short,
                                         lookback_hours=6, halflife_hours=3)

    # Long prefix: first k+50 bars, all news
    price_long  = price_df.iloc[:k + 50]
    res_long    = add_sentiment_features(price_long, news_df,
                                         lookback_hours=6, halflife_hours=3)

    # Compare settled indices
    violations_score = []
    violations_z     = []
    for i in range(settled):
        vs = res_short["sent_score"].iloc[i]
        vl = res_long["sent_score"].iloc[i]
        both_nan = np.isnan(vs) and np.isnan(vl)
        if not both_nan:
            if np.isnan(vs) != np.isnan(vl) or not np.isclose(vs, vl, rtol=1e-9):
                violations_score.append((i, vs, vl))

        zs = res_short["sent_z"].iloc[i]
        zl = res_long["sent_z"].iloc[i]
        both_nan_z = np.isnan(zs) and np.isnan(zl)
        if not both_nan_z:
            if np.isnan(zs) != np.isnan(zl) or not np.isclose(zs, zl, rtol=1e-9):
                violations_z.append((i, zs, zl))

    assert not violations_score, (
        f"sent_score repaint at {len(violations_score)} settled bar(s). "
        f"First: idx={violations_score[0][0]}, short={violations_score[0][1]:.6f}, "
        f"long={violations_score[0][2]:.6f}"
    )
    assert not violations_z, (
        f"sent_z repaint at {len(violations_z)} settled bar(s). "
        f"First: idx={violations_z[0][0]}, short={violations_z[0][1]:.6f}, "
        f"long={violations_z[0][2]:.6f}"
    )

    print(f"[PASS] test_append_invariance — {settled} settled indices checked "
          f"(prefix k={k}, suffix k+50={k+50})")


# ── Test 3: empty-news safety ─────────────────────────────────────────────────

def test_empty_news_no_crash():
    """
    Passing an empty news DataFrame must:
    - Not raise any exception.
    - Return sent_score filled with 0.0.
    - Return sent_count filled with 0.
    - Return sent_z filled with NaN (no data to z-score).
    """
    price_df = _make_price_df(n=50)
    news_df  = _empty_news_df()

    result = add_sentiment_features(price_df, news_df)

    assert "sent_score" in result.columns, "sent_score column missing"
    assert "sent_count" in result.columns, "sent_count column missing"
    assert "sent_z"     in result.columns, "sent_z column missing"

    assert (result["sent_score"] == 0.0).all(), (
        "sent_score should be 0.0 when news is empty"
    )
    assert (result["sent_count"] == 0).all(), (
        "sent_count should be 0 when news is empty"
    )
    assert result["sent_z"].isna().all(), (
        "sent_z should be NaN when news is empty"
    )
    assert len(result) == len(price_df), (
        "result must have the same number of rows as price_df"
    )

    print(f"[PASS] test_empty_news_no_crash — {len(price_df)} bars, "
          f"all zeros / NaN as expected")


# ── Test 4: None news safety ──────────────────────────────────────────────────

def test_none_news_no_crash():
    """Passing None as news_df must behave identically to empty DataFrame."""
    price_df = _make_price_df(n=30)

    result = add_sentiment_features(price_df, None)

    assert (result["sent_score"] == 0.0).all(), "sent_score != 0 for None news"
    assert (result["sent_count"] == 0).all(),   "sent_count != 0 for None news"
    assert result["sent_z"].isna().all(),        "sent_z not NaN for None news"

    print(f"[PASS] test_none_news_no_crash — {len(price_df)} bars checked")


# ── Test 5: sent_count accuracy ───────────────────────────────────────────────

def test_sent_count_accuracy():
    """
    sent_count at bar t must equal exactly the number of news items
    in the causal window (bar_ts - lookback_hours, bar_ts].
    Tested on a hand-crafted small example where the count is obvious.
    """
    # 10 hourly price bars
    price_idx = pd.date_range("2024-01-01 10:00", periods=10, freq="1h", tz="UTC")
    price_df  = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.5, "volume": 1000.0, "atr": 1.0},
        index=price_idx,
    )

    # 5 news items placed at known timestamps
    news_times = pd.DatetimeIndex([
        "2024-01-01 09:00",  # 1h before bar[0] (10:00) — in 3h window of bar[0]
        "2024-01-01 09:30",  # 30m before bar[0] — in window
        "2024-01-01 10:00",  # exactly at bar[0] — in window (ts <= t)
        "2024-01-01 11:00",  # at bar[1] (11:00) — NOT in bar[0] window
        "2024-01-01 12:15",  # after bar[1]
    ], tz="UTC")
    news_df = pd.DataFrame(
        {
            "title":  ["a", "b", "c", "d", "e"],
            "domain": ["x.com"] * 5,
            "tone":   [1.0, -1.0, 0.5, 2.0, -0.5],
            "url":    ["http://x.com"] * 5,
        },
        index=news_times,
    )

    result = add_sentiment_features(price_df, news_df, lookback_hours=3, halflife_hours=1)

    # bar[0] at 10:00 with 3h lookback: window is (07:00, 10:00]
    # items at 09:00, 09:30, 10:00 qualify  => count = 3
    assert result["sent_count"].iloc[0] == 3, (
        f"Expected sent_count=3 at bar[0], got {result['sent_count'].iloc[0]}"
    )
    # bar[1] at 11:00 with 3h lookback: window is (08:00, 11:00]
    # items at 09:00, 09:30, 10:00, 11:00 qualify => count = 4
    assert result["sent_count"].iloc[1] == 4, (
        f"Expected sent_count=4 at bar[1], got {result['sent_count'].iloc[1]}"
    )

    print(f"[PASS] test_sent_count_accuracy — sent_count correct at bar[0] and bar[1]")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_no_lookahead()
    test_append_invariance()
    test_empty_news_no_crash()
    test_none_news_no_crash()
    test_sent_count_accuracy()
    print("\nAll sentiment feature causality tests passed.")
