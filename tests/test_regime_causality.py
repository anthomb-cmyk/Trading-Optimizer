"""
tests/test_regime_causality.py
==============================
Causality (no look-ahead / no-repaint) tests for the standalone regime-filter
helpers in strategies/regime.py.

Dependencies: numpy, pandas only — no ta / pytest / vectorbt required at test
time (strategies.regime imports `ta`, but the test logic itself does not).

Run directly::

    python3 tests/test_regime_causality.py

Or via pytest.

Key property: append-invariance
-------------------------------
For every output Series produced by the regime helpers, computing on the
truncated frame df[:k] must give the same values as computing on the longer
frame df[:k+50], for every settled index t (t < k - WINDOW). Adding future
bars must not change any already-emitted value.

WINDOW is the largest lookback used by the helpers (the atr_percentile window,
default 100). Using `t < k - WINDOW` as the settled zone is intentionally
conservative — it clears every indicator's warmup with room to spare.
"""
import os
import sys

# Support direct execution from repo root OR from the tests/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from strategies.regime import (
    adx,
    atr_percentile,
    regime_label,
    regime_ok,
    REGIME_LABELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic OHLC builder (numpy/pandas only)
# ─────────────────────────────────────────────────────────────────────────────

SEED = 11
N_BARS = 700
CUTPOINT = 400      # k — split point well inside N_BARS
EXTRA_BARS = 50     # future bars appended for the invariance comparison
WINDOW = 100        # largest helper lookback (atr_percentile default window)


def _make_df(n: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """
    Reproducible synthetic OHLC frame with lowercase columns high/low/close
    (plus open for completeness). Built to contain both trending and ranging
    stretches and varying volatility, so all four regime labels appear.
    """
    rng = np.random.default_rng(seed)

    # Alternate drift to create trend and range stretches.
    drift = np.zeros(n)
    seg = 70
    for i in range(0, n, seg):
        d = rng.choice([-0.25, 0.0, 0.25, 0.6, -0.6])
        drift[i:i + seg] = d

    # Time-varying volatility so atr_percentile spans [0, 1].
    vol = 0.2 + 0.6 * (0.5 + 0.5 * np.sin(np.arange(n) / 40.0))
    returns = drift + rng.standard_normal(n) * vol
    close = 100.0 + np.cumsum(returns)

    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    wick = np.abs(rng.standard_normal(n)) * vol
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared invariance assertion
# ─────────────────────────────────────────────────────────────────────────────

def _assert_invariant(name: str, short_s: pd.Series, long_s: pd.Series,
                      settled_end: int) -> None:
    """Assert short_s[:settled_end] == long_s[:settled_end] element-wise."""
    assert len(short_s) >= settled_end and len(long_s) >= settled_end, (
        f"[{name}] series too short for settled_end={settled_end}"
    )
    for t in range(settled_end):
        v_short = short_s.iloc[t]
        v_long = long_s.iloc[t]
        # Treat both-NaN as equal (NaN == NaN is False in pandas).
        if _is_nan(v_short) and _is_nan(v_long):
            continue
        assert v_short == v_long, (
            f"[{name}] repaint at index {t}: short={v_short!r}, long={v_long!r}"
        )


def _is_nan(x) -> bool:
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — adx append-invariance
# ─────────────────────────────────────────────────────────────────────────────

def test_adx_causality():
    df = _make_df()
    k = CUTPOINT
    settled_end = k - WINDOW

    short_s = adx(df.iloc[:k].copy())
    long_s = adx(df.iloc[:k + EXTRA_BARS].copy())

    _assert_invariant("adx", short_s, long_s, settled_end)
    print(f"[PASS] test_adx_causality — {settled_end} settled indices stable")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — atr_percentile append-invariance + range bound [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_atr_percentile_causality():
    df = _make_df()
    k = CUTPOINT
    settled_end = k - WINDOW

    short_s = atr_percentile(df.iloc[:k].copy(), window=WINDOW)
    long_s = atr_percentile(df.iloc[:k + EXTRA_BARS].copy(), window=WINDOW)

    _assert_invariant("atr_percentile", short_s, long_s, settled_end)

    # Bound check on the long series: every non-NaN value in [0, 1].
    vals = long_s.dropna().values
    assert vals.size > 0, "atr_percentile produced no valid values"
    assert np.all(vals >= 0.0) and np.all(vals <= 1.0), (
        f"atr_percentile out of [0,1]: min={vals.min()}, max={vals.max()}"
    )
    print(f"[PASS] test_atr_percentile_causality — {settled_end} settled "
          f"indices stable, values in [0,1]")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — regime_label append-invariance + label domain
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_label_causality():
    df = _make_df()
    k = CUTPOINT
    settled_end = k - WINDOW

    short_s = regime_label(df.iloc[:k].copy())
    long_s = regime_label(df.iloc[:k + EXTRA_BARS].copy())

    _assert_invariant("regime_label", short_s, long_s, settled_end)

    # Every emitted label must be one of the four valid strings.
    produced = set(long_s.unique())
    assert produced <= REGIME_LABELS, (
        f"unexpected regime labels: {produced - REGIME_LABELS}"
    )
    print(f"[PASS] test_regime_label_causality — {settled_end} settled indices "
          f"stable, labels={sorted(produced)}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — regime_ok append-invariance + default/allow semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_ok_causality():
    df = _make_df()
    k = CUTPOINT
    settled_end = k - WINDOW
    allow = {"trend_hi", "trend_lo"}

    short_s = regime_ok(df.iloc[:k].copy(), allow=allow)
    long_s = regime_ok(df.iloc[:k + EXTRA_BARS].copy(), allow=allow)

    _assert_invariant("regime_ok", short_s, long_s, settled_end)

    assert short_s.dtype == bool, f"regime_ok must be bool, got {short_s.dtype}"

    # Default allow = all labels -> gate is always True.
    assert bool(regime_ok(df).all()), "default regime_ok must be all True"

    # regime_ok must agree with membership of regime_label in `allow`.
    labels = regime_label(df.iloc[:k].copy())
    expected = labels.isin(allow)
    assert short_s.equals(expected.rename("regime_ok")), (
        "regime_ok disagrees with regime_label membership"
    )
    print(f"[PASS] test_regime_ok_causality — {settled_end} settled indices "
          f"stable, allow={sorted(allow)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — supports both pytest and direct execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running regime causality tests...\n")
    test_adx_causality()
    test_atr_percentile_causality()
    test_regime_label_causality()
    test_regime_ok_causality()
    print("\nAll regime causality tests passed.")
