"""
strategies/sentiment_feature.py
================================
Causal sentiment feature engineering for the APEX Optimizer.

Public API
----------
    add_sentiment_features(
        price_df,
        news_df,
        score_col="tone",
        lookback_hours=12,
        halflife_hours=6,
    ) -> pd.DataFrame

    Returns a *copy* of ``price_df`` with three STRICTLY CAUSAL columns appended:

    sent_score : float
        Exponentially time-decayed mean of ``news_df[score_col]`` whose
        timestamp ts <= bar_ts, within the trailing ``lookback_hours`` window.
        Decay weight for a news item at lag L (hours) = exp(-ln(2) * L / halflife_hours).
        NaN when no qualifying news exists in the window.

    sent_count : int
        Number of news items whose ts falls in (bar_ts - lookback_hours, bar_ts].

    sent_z : float
        Rolling z-score of ``sent_score`` (trailing 20-bar window, min_periods=5).
        NaN when ``sent_score`` is NaN or the window is too small.

Causality contract
------------------
For bar at time ``t``, ONLY news rows with ``ts <= t`` (and ts >= t - lookback_hours)
are used.  The implementation uses ``np.searchsorted`` to locate the causal slice —
zero future news ever enters a bar's feature value.

Empty-news safety
-----------------
If ``news_df`` is empty (or None), all three columns are filled with 0 / 0 / NaN
respectively and the function never raises.

Efficiency
----------
The inner aggregation loop is O(n_bars * k_news_per_window) in the worst case but
the sorted ``searchsorted`` + vectorised numpy ops keep constants very small for
typical news volumes (~250 records per lookback window).  For very large news sets
a merge_asof + groupby path would be faster; this is left as a future optimisation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── Public function ───────────────────────────────────────────────────────────

def add_sentiment_features(
    price_df: pd.DataFrame,
    news_df: Optional[pd.DataFrame],
    score_col: str = "tone",
    lookback_hours: float = 12.0,
    halflife_hours: float = 6.0,
) -> pd.DataFrame:
    """
    Attach causal sentiment features to ``price_df``.

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV price frame with a tz-aware UTC DatetimeIndex.
    news_df : pd.DataFrame or None
        News frame with a UTC DatetimeIndex (``seendate``) and at least
        a ``score_col`` column (float).  Must be sorted ascending.
        Pass an empty DataFrame or ``None`` to get zero-filled columns.
    score_col : str, default "tone"
        Column in ``news_df`` to treat as the raw sentiment score.
    lookback_hours : float, default 12
        How many hours of news history to consider per bar (trailing window).
    halflife_hours : float, default 6
        Half-life for exponential time-decay of news scores.
        A score at exactly bar_ts receives weight 1.0.
        A score at bar_ts - halflife_hours receives weight 0.5.

    Returns
    -------
    pd.DataFrame
        Copy of ``price_df`` with columns ``sent_score``, ``sent_count``,
        and ``sent_z`` appended.  No look-ahead: only news with ts <= bar_ts
        is ever used.
    """
    out = price_df.copy()

    # ── Handle degenerate input ───────────────────────────────────────────────
    if news_df is None or news_df.empty or score_col not in (news_df.columns if news_df is not None else []):
        out["sent_score"] = 0.0
        out["sent_count"] = 0
        out["sent_z"]     = np.nan
        return out

    # ── Normalise timezone on both sides ─────────────────────────────────────
    # price index
    price_idx = price_df.index
    if price_idx.tz is None:
        price_idx = price_idx.tz_localize("UTC")
    else:
        price_idx = price_idx.tz_convert("UTC")

    # news index: sort ascending (required by searchsorted)
    news_idx = news_df.index
    if news_idx.tz is None:
        news_idx = news_idx.tz_localize("UTC")
    else:
        news_idx = news_idx.tz_convert("UTC")

    # Work with numpy int64 nanosecond timestamps for speed
    price_ns = price_idx.asi8              # shape (n_bars,)
    news_ns  = news_idx.asi8              # shape (n_news,)

    # Drop NaN scores; keep only rows where score is usable
    scores_raw = pd.to_numeric(news_df[score_col], errors="coerce").values.astype(float)
    valid_mask = ~np.isnan(scores_raw)

    # Sorted news arrays (already sorted ascending by construction; be safe)
    sort_order = np.argsort(news_ns, kind="stable")
    news_ns_s  = news_ns[sort_order]
    scores_s   = scores_raw[sort_order]
    valid_s    = valid_mask[sort_order]

    lookback_ns = int(lookback_hours * 3_600_000_000_000)  # hours -> nanoseconds
    ln2         = np.log(2.0)
    hl_ns       = halflife_hours * 3_600_000_000_000       # half-life in nanoseconds

    n_bars = len(price_ns)
    sent_score_arr = np.full(n_bars, np.nan, dtype=float)
    sent_count_arr = np.zeros(n_bars, dtype=np.int64)

    for i in range(n_bars):
        bar_ts = price_ns[i]
        window_start = bar_ts - lookback_ns

        # Causal slice: news with ts <= bar_ts AND ts > bar_ts - lookback_hours
        # searchsorted is O(log n) on the sorted news_ns_s array
        lo = int(np.searchsorted(news_ns_s, window_start + 1, side="left"))   # exclusive lower
        hi = int(np.searchsorted(news_ns_s, bar_ts,           side="right"))  # inclusive upper

        if hi <= lo:
            # No news in the causal window
            sent_count_arr[i] = 0
            continue

        # Only count rows with valid (non-NaN) scores
        slice_valid  = valid_s[lo:hi]
        slice_scores = scores_s[lo:hi]
        slice_ts     = news_ns_s[lo:hi]

        n_valid = int(slice_valid.sum())
        sent_count_arr[i] = hi - lo   # total items in window (including NaN-score rows)

        if n_valid == 0:
            continue  # sent_score stays NaN

        v_scores = slice_scores[slice_valid]
        v_ts     = slice_ts[slice_valid]

        # Lag in nanoseconds (bar_ts - article_ts >= 0 by construction)
        lag_ns  = (bar_ts - v_ts).astype(float)
        weights = np.exp(-ln2 * lag_ns / hl_ns)

        # Exponentially weighted mean
        w_sum = weights.sum()
        if w_sum > 0.0:
            sent_score_arr[i] = float(np.dot(weights, v_scores) / w_sum)
        # else remains NaN

    out["sent_score"] = sent_score_arr
    out["sent_count"] = sent_count_arr

    # ── Rolling z-score of sent_score (causal: only past values) ─────────────
    ss = out["sent_score"]
    roll_mean = ss.rolling(window=20, min_periods=5).mean()
    roll_std  = ss.rolling(window=20, min_periods=5).std()
    # Avoid division by zero
    roll_std_safe = roll_std.where(roll_std > 1e-10)
    out["sent_z"] = (ss - roll_mean) / roll_std_safe

    return out
