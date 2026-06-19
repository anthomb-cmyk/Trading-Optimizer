"""
tests/test_causality.py
=======================
Causality (no-look-ahead) regression tests for swing pivot detection.

Dependencies: numpy, pandas only — no vectorbt / numba / ta required.

Two guarantees verified
-----------------------
1. No-repaint guarantee
   Computing swings on df[:k] and on df[:k+50] produces IDENTICAL flags
   for every index < k - lb - 1.  Adding future bars must not change any
   already-emitted flag.

2. Pure-causality check
   A flag at index t must be unchanged when all bars after t are removed
   from the series.  If flag[t] == True on the full series it must still
   be True on the truncated series df[:t+1], and if it is False it must
   remain False.
"""
import sys
import os

# Allow running directly with `python3 tests/test_causality.py` from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from data.swings import _swing_highs, _swing_lows

# ---------------------------------------------------------------------------
# Synthetic OHLC helper
# ---------------------------------------------------------------------------
SEED = 42
N_BARS = 500
LOOKBACK = 10   # mirrors typical SMC["swing_lookback"]


def _make_ohlc(n: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """Build a reproducible synthetic OHLC DataFrame from a seeded random walk."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    noise_h = rng.uniform(0.1, 0.8, n)
    noise_l = rng.uniform(0.1, 0.8, n)
    noise_o = rng.uniform(-0.3, 0.3, n)

    high  = close + noise_h
    low   = close - noise_l
    open_ = close + noise_o
    # Ensure OHLC consistency
    high  = np.maximum(high, np.maximum(open_, close))
    low   = np.minimum(low,  np.minimum(open_, close))

    idx = pd.date_range("2020-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Test 1 — No-repaint guarantee
# ---------------------------------------------------------------------------
def test_no_repaint():
    """
    Flags at indices < k - lb - 1 must be identical whether computed on
    df[:k] or df[:k+50].  Adding 50 future bars must not alter any flag
    that was already settled.
    """
    lb = LOOKBACK
    df = _make_ohlc()
    k = 300   # cutpoint — well inside the 500-bar series

    highs_full  = df["high"].values
    lows_full   = df["low"].values

    # Compute on the shorter prefix
    sh_short = _swing_highs(highs_full[:k],      lb)
    sl_short = _swing_lows(lows_full[:k],        lb)

    # Compute on prefix + 50 extra bars
    sh_long  = _swing_highs(highs_full[:k + 50], lb)
    sl_long  = _swing_lows(lows_full[:k + 50],   lb)

    # Settled zone: indices strictly before k - lb - 1
    # (the lb-bar confirmation window can't be affected by bars > k - lb - 1)
    settled_end = k - lb - 1

    for t in range(settled_end):
        assert sh_short[t] == sh_long[t], (
            f"swing_high repaint at index {t}: "
            f"short={sh_short[t]}, long={sh_long[t]}"
        )
        assert sl_short[t] == sl_long[t], (
            f"swing_low repaint at index {t}: "
            f"short={sl_short[t]}, long={sl_long[t]}"
        )

    print(f"[PASS] test_no_repaint — {settled_end} settled indices checked, "
          f"swing highs={sh_short[:settled_end].sum()}, "
          f"swing lows={sl_short[:settled_end].sum()}")


# ---------------------------------------------------------------------------
# Test 2 — Pure causality: truncating after index t must not change flag[t]
# ---------------------------------------------------------------------------
def test_pure_causality():
    """
    For every bar t in [0, N_BARS), the flag at t computed on the full
    series must equal the flag at t computed on the series truncated at t+1.
    No future bar should influence the decision at t.
    """
    lb = LOOKBACK
    df = _make_ohlc()

    highs = df["high"].values
    lows  = df["low"].values

    sh_full = _swing_highs(highs, lb)
    sl_full = _swing_lows(lows,  lb)

    # We only need to check indices where the full series actually emits a
    # flag (True), plus a sample of False indices.  For efficiency, check all
    # True positions and every 5th False position.
    true_sh = np.where(sh_full)[0]
    true_sl = np.where(sl_full)[0]

    for t in true_sh:
        sh_trunc = _swing_highs(highs[: t + 1], lb)
        assert len(sh_trunc) == t + 1
        assert sh_trunc[t] == sh_full[t], (
            f"swing_high causality violated at t={t}: "
            f"full={sh_full[t]}, truncated={sh_trunc[t]}"
        )

    for t in true_sl:
        sl_trunc = _swing_lows(lows[: t + 1], lb)
        assert len(sl_trunc) == t + 1
        assert sl_trunc[t] == sl_full[t], (
            f"swing_low causality violated at t={t}: "
            f"full={sl_full[t]}, truncated={sl_trunc[t]}"
        )

    # Also spot-check a sample of False positions (every 5th bar, skip last lb)
    for t in range(0, len(highs) - lb, 5):
        if not sh_full[t]:
            sh_trunc = _swing_highs(highs[: t + 1], lb)
            assert not sh_trunc[t], (
                f"swing_high spurious True at t={t} after truncation"
            )
        if not sl_full[t]:
            sl_trunc = _swing_lows(lows[: t + 1], lb)
            assert not sl_trunc[t], (
                f"swing_low spurious True at t={t} after truncation"
            )

    print(f"[PASS] test_pure_causality — verified {len(true_sh)} sh pivots, "
          f"{len(true_sl)} sl pivots, plus sampled False indices")


# ---------------------------------------------------------------------------
# Test 3 — swing_high_price / swing_low_price columns
# ---------------------------------------------------------------------------
def test_swing_price_columns():
    """
    Verify that swing_high_price / swing_low_price:
    1. At every flag bar t, the recovered pivot price equals high[t - lb]
       AND high[t - lb] is the maximum of high[t - 2*lb : t + 1] (i.e. it
       really is the pivot).  Symmetric check for lows (minimum).
    2. No-repaint: recomputing on df[:k] vs df[:k+50] gives identical price
       values for indices < k - lb - 1 (treating NaN == NaN as equal).
    """
    lb = LOOKBACK
    df = _make_ohlc()
    high = df["high"]
    low  = df["low"]

    # Compute flag arrays
    swing_high = pd.Series(_swing_highs(high.values, lb), index=df.index)
    swing_low  = pd.Series(_swing_lows(low.values,  lb), index=df.index)

    # Build price columns the same way loader.py does
    sh_price = high.shift(lb).where(swing_high)
    sl_price = low.shift(lb).where(swing_low)

    # --- Guarantee 1: price correctness at every flagged bar -----------------
    flag_sh_indices = np.where(swing_high.values)[0]
    flag_sl_indices = np.where(swing_low.values)[0]

    for t in flag_sh_indices:
        recovered = sh_price.iloc[t]
        assert not np.isnan(recovered), f"swing_high_price is NaN at flag bar t={t}"
        # The recovered price must equal high[t - lb]
        assert recovered == high.iloc[t - lb], (
            f"swing_high_price[{t}] = {recovered} != high[{t - lb}] = {high.iloc[t - lb]}"
        )
        # high[t - lb] must be the maximum of high in [t-2*lb : t+1]
        window_start = max(0, t - 2 * lb)
        window_max   = high.iloc[window_start : t + 1].max()
        assert recovered == window_max, (
            f"high[{t - lb}] = {recovered} is not the window max {window_max} "
            f"in window [{window_start}:{t + 1}]"
        )

    for t in flag_sl_indices:
        recovered = sl_price.iloc[t]
        assert not np.isnan(recovered), f"swing_low_price is NaN at flag bar t={t}"
        assert recovered == low.iloc[t - lb], (
            f"swing_low_price[{t}] = {recovered} != low[{t - lb}] = {low.iloc[t - lb]}"
        )
        window_start = max(0, t - 2 * lb)
        window_min   = low.iloc[window_start : t + 1].min()
        assert recovered == window_min, (
            f"low[{t - lb}] = {recovered} is not the window min {window_min} "
            f"in window [{window_start}:{t + 1}]"
        )

    # --- Guarantee 2: no-repaint on price columns ----------------------------
    k = 300
    settled_end = k - lb - 1

    df_short = df.iloc[:k].copy()
    df_long  = df.iloc[:k + 50].copy()

    sh_short = pd.Series(_swing_highs(df_short["high"].values, lb), index=df_short.index)
    sl_short = pd.Series(_swing_lows(df_short["low"].values,   lb), index=df_short.index)
    sh_long  = pd.Series(_swing_highs(df_long["high"].values,  lb), index=df_long.index)
    sl_long  = pd.Series(_swing_lows(df_long["low"].values,    lb), index=df_long.index)

    sh_price_short = df_short["high"].shift(lb).where(sh_short)
    sl_price_short = df_short["low"].shift(lb).where(sl_short)
    sh_price_long  = df_long["high"].shift(lb).where(sh_long)
    sl_price_long  = df_long["low"].shift(lb).where(sl_long)

    for t in range(settled_end):
        v_short = sh_price_short.iloc[t]
        v_long  = sh_price_long.iloc[t]
        both_nan = np.isnan(v_short) and np.isnan(v_long)
        assert both_nan or (v_short == v_long), (
            f"swing_high_price no-repaint violated at t={t}: "
            f"short={v_short}, long={v_long}"
        )
        v_short = sl_price_short.iloc[t]
        v_long  = sl_price_long.iloc[t]
        both_nan = np.isnan(v_short) and np.isnan(v_long)
        assert both_nan or (v_short == v_long), (
            f"swing_low_price no-repaint violated at t={t}: "
            f"short={v_short}, long={v_long}"
        )

    print(f"[PASS] test_swing_price_columns — verified {len(flag_sh_indices)} sh pivots, "
          f"{len(flag_sl_indices)} sl pivots; no-repaint checked over {settled_end} indices")


# ---------------------------------------------------------------------------
# Entry point for both pytest and direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_no_repaint()
    test_pure_causality()
    test_swing_price_columns()
    print("\nAll causality tests passed.")
