"""
tests/test_detector_causality.py
=================================
Causality (no look-ahead / no-repaint) regression tests for the shared SMC
detectors in strategies/detectors.py.

Dependencies: numpy, pandas only — no vectorbt / numba / pytest required.

Run directly::

    python3 tests/test_detector_causality.py

Or via pytest (both modes are supported).

Append-invariance test (the key property)
-----------------------------------------
For each detector, we verify that computing on df[:k] gives the same output
as computing on df[:k+50] for all indices strictly before k - margin.
Adding future bars must NOT alter any already-emitted flag or value.

Detectors tested
----------------
detect_fvg          — 3-bar FVG (causal as-is)
detect_fvg_zones    — FVG zone bounds (causal as-is)
detect_order_blocks — OB via displacement (FIXED in T1.2)
detect_bos          — Break of Structure (causal as-is)
detect_choch        — Change of Character (causal as-is)
"""
import sys
import os

# Support direct execution from repo root OR from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from strategies.detectors import (
    detect_fvg,
    detect_fvg_zones,
    detect_order_blocks,
    detect_bos,
    detect_choch,
)
from data.swings import _swing_highs, _swing_lows


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic OHLC builder
# ─────────────────────────────────────────────────────────────────────────────

SEED   = 42
N_BARS = 600
SWING_LB = 10   # mirrors SMC["swing_lookback"]


def _make_df(n: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """
    Build a reproducible synthetic OHLC DataFrame with all indicator columns
    required by the five detectors:

    Required columns:
      high, low, open, close, atr, atr_avg, body, range, bullish,
      swing_high_price, swing_low_price

    Candle construction note
    ------------------------
    To ensure the OB detector has real hits to test (body_pct >= 0.6 AND
    range >= ATR * 0.8), we model each bar as either a "strong" impulse candle
    (small wicks, large body) with 20% probability, or a doji/indecision candle
    otherwise.  Strong candles alternate direction randomly.  This gives ~40-60
    displacement candles in a 600-bar series — enough for a meaningful test
    without forcing artificial results.
    """
    rng = np.random.default_rng(seed)

    # Price random walk with moderate drift
    returns   = rng.standard_normal(n) * 0.4
    close     = 100.0 + np.cumsum(returns)

    # Candle type: True = strong impulse, False = indecision/doji
    is_strong = rng.random(n) < 0.20

    # Body direction: +1 bullish, -1 bearish
    direction = np.where(rng.random(n) < 0.5, 1.0, -1.0)

    # Strong candle: body is 70-95% of the total range; small wicks
    # Indecision candle: body is 5-35% of range; random wick distribution
    total_range = np.where(is_strong,
                           rng.uniform(0.4, 1.0, n),   # strong: large range
                           rng.uniform(0.1, 0.5, n))   # doji: smaller range

    body_frac   = np.where(is_strong,
                           rng.uniform(0.70, 0.95, n), # strong: big body
                           rng.uniform(0.05, 0.35, n)) # doji: small body

    body_size   = total_range * body_frac

    # Build open / close from close and direction
    open_  = np.where(direction > 0,
                      close - body_size,  # bullish: open < close
                      close + body_size)  # bearish: open > close

    wick_upper = total_range * rng.uniform(0.02, 0.15, n)
    wick_lower = total_range * rng.uniform(0.02, 0.15, n)

    high_ = np.maximum(open_, close) + wick_upper
    low_  = np.minimum(open_, close) - wick_lower

    # Enforce OHLC consistency (guarantees)
    high_ = np.maximum(high_, np.maximum(open_, close))
    low_  = np.minimum(low_,  np.minimum(open_, close))

    idx = pd.date_range("2020-01-01", periods=n, freq="15min")

    df = pd.DataFrame(
        {"open": open_, "high": high_, "low": low_, "close": close},
        index=idx,
    )

    # ── Derived indicator columns ────────────────────────────────────────────

    # ATR: simple 14-bar rolling average of true range (pure-pandas, no TA lib)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"]     = tr.rolling(14, min_periods=1).mean()
    df["atr_avg"] = df["atr"].rolling(50, min_periods=1).mean()

    # Candle body and range
    df["range"]    = df["high"] - df["low"]
    df["body"]     = (df["close"] - df["open"]).abs()
    df["bullish"]  = df["close"] >= df["open"]

    # Swing high / low prices (causal, from data/swings.py)
    sh_flags = _swing_highs(df["high"].values, SWING_LB)
    sl_flags = _swing_lows(df["low"].values,   SWING_LB)

    sh_series = pd.Series(sh_flags, index=df.index)
    sl_series = pd.Series(sl_flags, index=df.index)

    # swing_high_price: the actual high at the pivot (lb bars before the flag)
    df["swing_high_price"] = df["high"].shift(SWING_LB).where(sh_series)
    df["swing_low_price"]  = df["low"].shift(SWING_LB).where(sl_series)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Shared append-invariance check
# ─────────────────────────────────────────────────────────────────────────────

CUTPOINT   = 350   # k — split point well inside N_BARS
EXTRA_BARS = 50    # how many future bars to append
MARGIN     = 25    # conservative settled zone: indices < k - margin


def _check_series_invariant(
    name: str,
    short_s: pd.Series,
    long_s: pd.Series,
    settled_end: int,
) -> None:
    """Assert that short_s[:settled_end] == long_s[:settled_end] (element-wise)."""
    for t in range(settled_end):
        v_short = short_s.iloc[t]
        v_long  = long_s.iloc[t]
        # Treat both-NaN as equal (NaN == NaN is False in pandas)
        if pd.isna(v_short) and pd.isna(v_long):
            continue
        assert v_short == v_long, (
            f"[{name}] repaint at index {t}: "
            f"short={v_short}, long={v_long}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — detect_fvg
# ─────────────────────────────────────────────────────────────────────────────

def test_fvg_causality():
    """
    Append-invariance for detect_fvg.
    Verdict: causal as-is (uses shift(2) = past only).
    """
    df = _make_df()
    k  = CUTPOINT
    settled_end = k - MARGIN

    bull_short, bear_short = detect_fvg(df.iloc[:k].copy())
    bull_long,  bear_long  = detect_fvg(df.iloc[:k + EXTRA_BARS].copy())

    _check_series_invariant("detect_fvg[bull]", bull_short, bull_long, settled_end)
    _check_series_invariant("detect_fvg[bear]", bear_short, bear_long, settled_end)

    n_bull = bull_short.iloc[:settled_end].sum()
    n_bear = bear_short.iloc[:settled_end].sum()
    print(f"[PASS] test_fvg_causality — {settled_end} indices, "
          f"bull_fvg={n_bull}, bear_fvg={n_bear}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — detect_fvg_zones
# ─────────────────────────────────────────────────────────────────────────────

def test_fvg_zones_causality():
    """
    Append-invariance for detect_fvg_zones (top/bottom/mid columns).
    Verdict: causal as-is.
    """
    df = _make_df()
    k  = CUTPOINT
    settled_end = k - MARGIN

    bull_short, bear_short = detect_fvg_zones(df.iloc[:k].copy())
    bull_long,  bear_long  = detect_fvg_zones(df.iloc[:k + EXTRA_BARS].copy())

    for col in ("top", "bottom", "mid"):
        _check_series_invariant(
            f"detect_fvg_zones[bull.{col}]",
            bull_short[col], bull_long[col], settled_end,
        )
        _check_series_invariant(
            f"detect_fvg_zones[bear.{col}]",
            bear_short[col], bear_long[col], settled_end,
        )

    print(f"[PASS] test_fvg_zones_causality — {settled_end} indices checked")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — detect_order_blocks  (the fixed detector)
# ─────────────────────────────────────────────────────────────────────────────

def test_order_blocks_causality():
    """
    Append-invariance for detect_order_blocks after the T1.2 fix.

    The old code used shift(-j) which caused repaint: OBs marked at bar i
    would change whenever bars i+1..i+lookback were later revealed.
    The fix places the flag at the displacement bar (i+j) so settled indices
    are stable regardless of what comes after.
    """
    df = _make_df()
    k  = CUTPOINT
    # OB needs up to `lookback` bars to confirm; use default lookback=20.
    # settled_end must clear that confirmation window.
    lookback    = 20
    settled_end = k - lookback - 5   # extra 5 as safety margin

    bull_short, bear_short, *_ = detect_order_blocks(df.iloc[:k].copy(), lookback=lookback)
    bull_long,  bear_long,  *_ = detect_order_blocks(
        df.iloc[:k + EXTRA_BARS].copy(), lookback=lookback
    )

    _check_series_invariant("detect_order_blocks[bull]", bull_short, bull_long, settled_end)
    _check_series_invariant("detect_order_blocks[bear]", bear_short, bear_long, settled_end)

    n_bull = bull_short.iloc[:settled_end].sum()
    n_bear = bear_short.iloc[:settled_end].sum()
    print(f"[PASS] test_order_blocks_causality — {settled_end} indices, "
          f"bull_ob={n_bull}, bear_ob={n_bear}")


def test_order_blocks_no_future_dependency():
    """
    Stronger check: for every True flag at index t in bull_ob / bear_ob,
    truncating the series at t+1 must yield the SAME True flag.
    (Pure causality: flag at t depends only on data at indices <= t.)
    """
    df = _make_df()
    lookback = 20

    bull_full, bear_full, *_ = detect_order_blocks(df, lookback=lookback)

    # Check every True bull_ob position
    for t in np.where(bull_full.values)[0]:
        sub = df.iloc[: t + 1].copy()
        bull_sub, *_ = detect_order_blocks(sub, lookback=lookback)
        assert len(bull_sub) == t + 1, f"length mismatch at t={t}"
        assert bull_sub.iloc[t] == bull_full.iloc[t], (
            f"detect_order_blocks[bull] causality violated at t={t}: "
            f"full={bull_full.iloc[t]}, truncated={bull_sub.iloc[t]}"
        )

    # Check every True bear_ob position
    for t in np.where(bear_full.values)[0]:
        sub = df.iloc[: t + 1].copy()
        _, bear_sub, *_ = detect_order_blocks(sub, lookback=lookback)
        assert len(bear_sub) == t + 1, f"length mismatch at t={t}"
        assert bear_sub.iloc[t] == bear_full.iloc[t], (
            f"detect_order_blocks[bear] causality violated at t={t}: "
            f"full={bear_full.iloc[t]}, truncated={bear_sub.iloc[t]}"
        )

    n_bull = bull_full.sum()
    n_bear = bear_full.sum()
    print(f"[PASS] test_order_blocks_no_future_dependency — "
          f"verified {n_bull} bull OBs, {n_bear} bear OBs")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3b — detect_order_blocks zone correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_order_block_zone_correctness():
    """
    Verify that the _high/_low zone arrays returned by detect_order_blocks:

    (a) At every bull_ob flag bar: _high and _low are non-NaN, _high >= _low,
        and both values equal the high/low of some bar within the prior
        lookback window (i.e. the OB candle, not the displacement bar).

    (b) Append-invariance of the zone values: for indices settled well before
        the cutpoint, running detect_order_blocks on df[:k] vs df[:k+50] yields
        the same bull_ob_high / bull_ob_low values.
    """
    df      = _make_df()
    lookback = 20

    bull_ob, bear_ob, bull_ob_high, bull_ob_low, bear_ob_high, bear_ob_low = detect_order_blocks(
        df, lookback=lookback
    )

    # ── (a) Zone sanity at every flag bar ─────────────────────────────────────
    high_vals = df["high"].values
    low_vals  = df["low"].values

    bull_flag_indices = np.where(bull_ob.values)[0]
    assert len(bull_flag_indices) > 0, "No bull OBs found — seed or df too short?"

    for t in bull_flag_indices:
        h = bull_ob_high.iloc[t]
        l = bull_ob_low.iloc[t]

        assert not pd.isna(h), f"bull_ob_high is NaN at flag bar t={t}"
        assert not pd.isna(l), f"bull_ob_low is NaN at flag bar t={t}"
        assert h >= l, f"bull_ob_high < bull_ob_low at t={t}: high={h}, low={l}"

        # The zone values must match the high/low of some bar within the prior
        # lookback window (the OB candle sits between t-lookback and t-1).
        window_start = max(0, t - lookback)
        window_highs = high_vals[window_start:t]
        window_lows  = low_vals[window_start:t]

        assert np.any(np.isclose(window_highs, h, rtol=1e-9)), (
            f"bull_ob_high={h} at t={t} does not match any bar high in "
            f"window [{window_start}, {t})"
        )
        assert np.any(np.isclose(window_lows, l, rtol=1e-9)), (
            f"bull_ob_low={l} at t={t} does not match any bar low in "
            f"window [{window_start}, {t})"
        )

    bear_flag_indices = np.where(bear_ob.values)[0]
    assert len(bear_flag_indices) > 0, "No bear OBs found — seed or df too short?"

    for t in bear_flag_indices:
        h = bear_ob_high.iloc[t]
        l = bear_ob_low.iloc[t]

        assert not pd.isna(h), f"bear_ob_high is NaN at flag bar t={t}"
        assert not pd.isna(l), f"bear_ob_low is NaN at flag bar t={t}"
        assert h >= l, f"bear_ob_high < bear_ob_low at t={t}: high={h}, low={l}"

        window_start = max(0, t - lookback)
        window_highs = high_vals[window_start:t]
        window_lows  = low_vals[window_start:t]

        assert np.any(np.isclose(window_highs, h, rtol=1e-9)), (
            f"bear_ob_high={h} at t={t} does not match any bar high in "
            f"window [{window_start}, {t})"
        )
        assert np.any(np.isclose(window_lows, l, rtol=1e-9)), (
            f"bear_ob_low={l} at t={t} does not match any bar low in "
            f"window [{window_start}, {t})"
        )

    # ── (b) Append-invariance of zone values ──────────────────────────────────
    k           = CUTPOINT
    settled_end = k - lookback - 5   # same margin as the bool test

    (_, _, bull_ob_high_short, bull_ob_low_short,
     bear_ob_high_short, bear_ob_low_short) = detect_order_blocks(
        df.iloc[:k].copy(), lookback=lookback
    )
    (_, _, bull_ob_high_long, bull_ob_low_long,
     bear_ob_high_long, bear_ob_low_long) = detect_order_blocks(
        df.iloc[:k + EXTRA_BARS].copy(), lookback=lookback
    )

    _check_series_invariant(
        "detect_order_blocks[bull_ob_high]",
        bull_ob_high_short, bull_ob_high_long, settled_end,
    )
    _check_series_invariant(
        "detect_order_blocks[bull_ob_low]",
        bull_ob_low_short, bull_ob_low_long, settled_end,
    )
    _check_series_invariant(
        "detect_order_blocks[bear_ob_high]",
        bear_ob_high_short, bear_ob_high_long, settled_end,
    )
    _check_series_invariant(
        "detect_order_blocks[bear_ob_low]",
        bear_ob_low_short, bear_ob_low_long, settled_end,
    )

    print(f"[PASS] test_order_block_zone_correctness — "
          f"{len(bull_flag_indices)} bull OBs, {len(bear_flag_indices)} bear OBs verified; "
          f"zone append-invariance confirmed to index {settled_end}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — detect_bos
# ─────────────────────────────────────────────────────────────────────────────

def test_bos_causality():
    """
    Append-invariance for detect_bos.
    Verdict: causal as-is — rolling().shift(+confirm_bars) only lags.
    """
    df = _make_df()
    k  = CUTPOINT
    lookback = 20
    # settled zone: indices before k - lookback - confirm_bars
    settled_end = k - lookback - 2

    bull_short, bear_short = detect_bos(df.iloc[:k].copy(), lookback=lookback)
    bull_long,  bear_long  = detect_bos(
        df.iloc[:k + EXTRA_BARS].copy(), lookback=lookback
    )

    _check_series_invariant("detect_bos[bull]", bull_short, bull_long, settled_end)
    _check_series_invariant("detect_bos[bear]", bear_short, bear_long, settled_end)

    n_bull = bull_short.iloc[:settled_end].sum()
    n_bear = bear_short.iloc[:settled_end].sum()
    print(f"[PASS] test_bos_causality — {settled_end} indices, "
          f"bull_bos={n_bull}, bear_bos={n_bear}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — detect_choch
# ─────────────────────────────────────────────────────────────────────────────

def test_choch_causality():
    """
    Append-invariance for detect_choch.
    Verdict: causal as-is — depends only on ffill'd past swing prices + shift(1).
    Note: swing prices are derived from _swing_highs/_swing_lows, themselves
    causal (confirmed at i+lb).  The settled zone must clear the lb warmup.
    """
    df = _make_df()
    k  = CUTPOINT
    # swing confirmation needs SWING_LB bars; add margin
    settled_end = k - SWING_LB - 5

    bull_short, bear_short = detect_choch(df.iloc[:k].copy())
    bull_long,  bear_long  = detect_choch(
        df.iloc[:k + EXTRA_BARS].copy()
    )

    _check_series_invariant("detect_choch[bull]", bull_short, bull_long, settled_end)
    _check_series_invariant("detect_choch[bear]", bear_short, bear_long, settled_end)

    n_bull = bull_short.iloc[:settled_end].sum()
    n_bear = bear_short.iloc[:settled_end].sum()
    print(f"[PASS] test_choch_causality — {settled_end} indices, "
          f"bull_choch={n_bull}, bear_choch={n_bear}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — supports both pytest and direct execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running detector causality tests...\n")
    test_fvg_causality()
    test_fvg_zones_causality()
    test_order_blocks_causality()
    test_order_blocks_no_future_dependency()
    test_order_block_zone_correctness()
    test_bos_causality()
    test_choch_causality()
    print("\nAll detector causality tests passed.")
