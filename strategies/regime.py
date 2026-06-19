"""
strategies/regime.py
=====================
Standalone, causal regime-filter helpers.

This module is NOT a BaseStrategy. It provides small, composable functions a
strategy (or the optimizer) can call to classify the current market regime and
gate signals accordingly.

Dependencies: numpy, pandas, ta (ta.trend.ADXIndicator only).

Expected input
--------------
A pandas DataFrame `df` with lowercase OHLC columns: "high", "low", "close"
(matching the convention used across strategies/ and data/).

Causality contract
-------------------
Every output value at bar t is a function of input rows with index <= t only.
No look-ahead, no repaint. Concretely:

    f(df.iloc[:k]).iloc[t] == f(df.iloc[:k + m]).iloc[t]

for any m >= 0, for all settled indices t (i.e. past the indicator warmup).
This is enforced by tests/test_regime_causality.py (append-invariance).

Functions
---------
adx(df, period=14)                       -> pd.Series  (ADX, Wilder)
atr_percentile(df, window=100)           -> pd.Series  rolling pctile in [0, 1]
regime_label(df, adx_trend=25, vol_hi=0.7) -> pd.Series of regime strings
regime_ok(df, allow=None)                -> pd.Series of bool

Regime labels
-------------
Two binary axes combined into four labels:
  - trend axis : adx >= adx_trend  -> "trend",  else "range"
  - vol axis   : atr_percentile >= vol_hi -> "hi", else "lo"

  {"trend_lo", "trend_hi", "range_lo", "range_hi"}
"""
from __future__ import annotations

from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: All valid regime labels. Used as the default `allow` set for regime_ok.
REGIME_LABELS: Set[str] = {"trend_lo", "trend_hi", "range_lo", "range_hi"}


# ─────────────────────────────────────────────────────────────────────────────
# Internal: causal True-Range / ATR (Wilder)
# ─────────────────────────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    Causal True Range.

    TR_t = max(high_t - low_t,
               |high_t - close_{t-1}|,
               |low_t  - close_{t-1}|)

    Uses close.shift(1) so TR_t depends only on bars t and t-1. The first bar
    has no prior close; TR there falls back to (high - low).
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)

    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()

    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    # First bar: no prev_close, so hc/lc are NaN -> max ignores NaN and yields hl.
    return tr


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Causal Average True Range using Wilder's smoothing, expressed as a
    plain recursive (causal) EMA so settled values are append-invariant.

    We use the simple causal form ATR = TR.ewm(alpha=1/period).mean(), which is
    a strictly backward-looking recursion: ATR_t depends only on TR_0..TR_t.
    """
    tr = _true_range(df)
    # adjust=False -> recursive Wilder-style smoothing, fully causal.
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX), Wilder.

    Causal: ADX_t uses only bars <= t. Built on ta.trend.ADXIndicator, which
    relies on Wilder smoothing (a backward-only recursion). The warmup region
    (roughly the first 2*period bars) is NaN / unsettled and must be excluded
    from any append-invariance comparison.

    Parameters
    ----------
    df : DataFrame with "high", "low", "close".
    period : Wilder lookback (default 14).

    Returns
    -------
    pd.Series named "adx", indexed like df.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    indicator = ADXIndicator(
        high=high,
        low=low,
        close=close,
        window=period,
        fillna=False,
    )
    out = indicator.adx()
    out.name = "adx"
    out.index = df.index
    return out


def atr_percentile(df: pd.DataFrame, window: int = 100) -> pd.Series:
    """
    Rolling percentile rank of ATR, in [0, 1].

    For each bar t, computes the fraction of ATR values in the trailing window
    (the last `window` bars, ending at and including t) that are <= ATR_t.

    Causal: only the trailing window up to t is used, so the value at t never
    changes when future bars are appended (once t is past the ATR warmup).

    Parameters
    ----------
    df : DataFrame with "high", "low", "close".
    window : trailing window length for the percentile rank (default 100).

    Returns
    -------
    pd.Series named "atr_percentile" with values in [0, 1] (NaN during warmup).
    """
    atr = _atr(df)

    def _pct_rank(arr: np.ndarray) -> float:
        # arr is the trailing window ending at the current bar.
        # Last element is the current ATR; rank it within the window.
        current = arr[-1]
        if np.isnan(current):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        return float(np.mean(valid <= current))

    out = atr.rolling(window=window, min_periods=1).apply(_pct_rank, raw=True)
    out.name = "atr_percentile"
    return out


def regime_label(
    df: pd.DataFrame,
    adx_trend: float = 25.0,
    vol_hi: float = 0.7,
) -> pd.Series:
    """
    Classify each bar into one of four regime labels by combining the ADX
    (trend vs range) and ATR-percentile (high vs low volatility) axes.

    trend axis : adx >= adx_trend       -> "trend", else "range"
    vol axis   : atr_percentile >= vol_hi -> "hi",   else "lo"

    Labels: {"trend_lo", "trend_hi", "range_lo", "range_hi"}.

    Causal: composed entirely from adx() and atr_percentile(), both causal.
    During the indicator warmup either input may be NaN; we treat NaN
    conservatively:
      - NaN adx  -> "range" (not enough evidence of a trend)
      - NaN vol  -> "lo"    (not enough evidence of high volatility)
    so a label is always defined, and settled labels are append-invariant.

    Returns
    -------
    pd.Series of str named "regime", indexed like df.
    """
    a = adx(df, period=14)
    v = atr_percentile(df, window=100)

    # Conservative NaN handling keeps labels causal AND always-defined.
    is_trend = (a >= adx_trend).fillna(False)
    is_hi = (v >= vol_hi).fillna(False)

    trend_part = np.where(is_trend.values, "trend", "range")
    vol_part = np.where(is_hi.values, "hi", "lo")

    labels = pd.Series(
        [f"{t}_{vv}" for t, vv in zip(trend_part, vol_part)],
        index=df.index,
        name="regime",
        dtype="object",
    )
    return labels


def regime_ok(
    df: pd.DataFrame,
    allow: Optional[Iterable[str]] = None,
) -> pd.Series:
    """
    Boolean gate: True where the bar's regime label is in `allow`.

    Parameters
    ----------
    df : DataFrame with "high", "low", "close".
    allow : iterable of permitted regime labels. Defaults to all four labels
            (i.e. the gate is always True), so a strategy can opt in.

    Returns
    -------
    pd.Series of bool named "regime_ok", indexed like df. Causal by
    construction (derived solely from regime_label).
    """
    if allow is None:
        allowed: Set[str] = set(REGIME_LABELS)
    else:
        allowed = set(allow)

    labels = regime_label(df)
    out = labels.isin(allowed)
    out.name = "regime_ok"
    return out
