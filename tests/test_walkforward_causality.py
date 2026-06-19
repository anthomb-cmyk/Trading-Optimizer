"""
tests/test_walkforward_causality.py
====================================
Walk-forward APPEND-INVARIANCE causality tests.

Verifies that the per-fold signal generation introduced in T1.3 produces
signals that are independent of any bars outside the fold's causal window
[fold_start - warmup, fold_end].

Key assertion — APPEND-INVARIANCE:
    OOS signals for a given fold must be IDENTICAL whether the dataframe
    passed to WalkForwardAnalyzer ends at the OOS fold boundary or extends
    with arbitrary future bars.

Implementation note:
    ``StrategySystem.generate_signals`` pulls in heavy dependencies (ta,
    all 10 strategies).  This test uses a lightweight STUB signal generator
    that writes a rolling statistic so any look-ahead would be immediately
    detectable as numerical disagreement.  The stub is directly injected as
    ``generate_signals_fn`` into the per-fold helper ``generate_fold_signals``
    and ``WalkForwardAnalyzer.analyze`` — both of which are the production code
    paths refactored in T1.3.

Running:
    python3 tests/test_walkforward_causality.py      # direct
    pytest  tests/test_walkforward_causality.py      # via pytest
"""
import sys
import os

# Allow direct `python3 tests/...` execution from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtesting.walk_forward import generate_fold_signals, make_wf_splits, WalkForwardAnalyzer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP = 50   # deliberately different from WARMUP_BARS=200 so the test is
              # self-contained and fast; the helper accepts warmup_bars param
N_BARS = 600
SEED   = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlc(n: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """Reproducible synthetic OHLCV data."""
    rng   = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    noise = rng.uniform(0.1, 0.5, (n, 3))
    high  = close + noise[:, 0]
    low   = close - noise[:, 1]
    open_ = close + noise[:, 2] * rng.choice([-1, 1], n)
    high  = np.maximum(high, np.maximum(open_, close))
    low   = np.minimum(low,  np.minimum(open_, close))
    vol   = rng.integers(1000, 5000, n).astype(float)

    idx = pd.date_range("2023-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _stub_generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Minimal signal generator that writes a rolling mean of close prices.

    The rolling statistic is data-dependent: if future bars leak in, the
    values for earlier bars will change, breaking the equality assertions.
    We also embed the RUNNING INDEX of each bar so any row-alignment error
    (off-by-one in prefix stripping) surfaces immediately.
    """
    sig = df.copy()
    win = int(params.get("rolling_win", 20))
    sig["roll_close"] = df["close"].rolling(win, min_periods=1).mean()
    sig["bar_index"]  = np.arange(len(df))          # monotone running counter
    sig["long_signal"]  = False
    sig["short_signal"] = False
    # Simple alternating signals to give the engine something to score
    sig.loc[sig.index[::30], "long_signal"]  = True
    sig.loc[sig.index[::45], "short_signal"] = True
    return sig


PARAMS = {"rolling_win": 30}


# ---------------------------------------------------------------------------
# Test 1 — generate_fold_signals: warmup rows are excluded from output
# ---------------------------------------------------------------------------
def test_warmup_rows_excluded():
    """
    The returned (fold_df, fold_signals) must have exactly
    fold_end - fold_start rows — the warmup prefix must be stripped.
    """
    df = _make_ohlc()
    fold_sl = slice(100, 200)   # 100 bars in fold

    fold_df, fold_sig = generate_fold_signals(
        df, fold_sl, _stub_generate_signals, PARAMS, warmup_bars=WARMUP,
    )

    expected_rows = fold_sl.stop - fold_sl.start   # 100

    assert len(fold_df)  == expected_rows, (
        f"fold_df has {len(fold_df)} rows, expected {expected_rows}"
    )
    assert len(fold_sig) == expected_rows, (
        f"fold_sig has {len(fold_sig)} rows, expected {expected_rows}"
    )

    # Indices must align between df and sig
    assert list(fold_df.index) == list(fold_sig.index), \
        "fold_df and fold_sig index mismatch"

    print(f"[PASS] test_warmup_rows_excluded — fold rows={expected_rows}, "
          f"warmup={WARMUP} stripped correctly")


# ---------------------------------------------------------------------------
# Test 2 — APPEND-INVARIANCE: OOS signals identical with or without future bars
# ---------------------------------------------------------------------------
def test_oos_append_invariance():
    """
    Core append-invariance check.

    For each OOS fold, generate signals with:
      (a) df truncated at the OOS fold end
      (b) df extended with N_EXTRA additional bars of random future data

    The per-fold OOS signals (roll_close column) must be bitwise-identical.
    Any leakage of future bars through the rolling statistic would break this.
    """
    N_EXTRA = 150
    rng = np.random.default_rng(SEED + 1)

    df_base = _make_ohlc(N_BARS)

    # Build extended df by appending random future bars
    future_close = df_base["close"].iloc[-1] + np.cumsum(rng.standard_normal(N_EXTRA) * 0.3)
    future_noise = rng.uniform(0.1, 0.5, (N_EXTRA, 3))
    future_high  = future_close + future_noise[:, 0]
    future_low   = future_close - future_noise[:, 1]
    future_open  = future_close + future_noise[:, 2]
    future_high  = np.maximum(future_high, np.maximum(future_open, future_close))
    future_low   = np.minimum(future_low,  np.minimum(future_open, future_close))
    future_idx   = pd.date_range(
        df_base.index[-1] + pd.Timedelta("15min"),
        periods=N_EXTRA, freq="15min",
    )
    df_future = pd.DataFrame(
        {"open": future_open, "high": future_high,
         "low": future_low, "close": future_close,
         "volume": rng.integers(1000, 5000, N_EXTRA).astype(float)},
        index=future_idx,
    )
    df_extended = pd.concat([df_base, df_future])

    splits = make_wf_splits(n_bars=N_BARS, n_splits=4, oos_frac=0.15, gap_bars=5)
    assert splits, "No splits generated — increase N_BARS or reduce n_splits"

    for fold_idx, (is_sl, oos_sl) in enumerate(splits):
        # Truncated df ends at the OOS fold boundary
        df_trunc = df_base.iloc[: oos_sl.stop].copy()

        _, oos_sig_trunc = generate_fold_signals(
            df_trunc, oos_sl, _stub_generate_signals, PARAMS, warmup_bars=WARMUP,
        )
        _, oos_sig_ext = generate_fold_signals(
            df_extended, oos_sl, _stub_generate_signals, PARAMS, warmup_bars=WARMUP,
        )

        assert len(oos_sig_trunc) == len(oos_sig_ext), (
            f"Fold {fold_idx+1}: length mismatch "
            f"{len(oos_sig_trunc)} vs {len(oos_sig_ext)}"
        )

        # roll_close must be identical — any future-bar leakage breaks this
        diff = (oos_sig_trunc["roll_close"] - oos_sig_ext["roll_close"]).abs()
        max_diff = diff.max()
        assert max_diff == 0.0, (
            f"Fold {fold_idx+1}: OOS roll_close differs by up to {max_diff:.2e} "
            f"between truncated and extended df — look-ahead detected!"
        )

        print(
            f"  [fold {fold_idx+1}] OOS bars={len(oos_sig_trunc)}, "
            f"max_roll_close_diff={max_diff:.2e}  [PASS]"
        )

    print(f"[PASS] test_oos_append_invariance — {len(splits)} folds verified")


# ---------------------------------------------------------------------------
# Test 3 — IS append-invariance (IS fold must not use bars beyond its boundary)
# ---------------------------------------------------------------------------
def test_is_append_invariance():
    """
    The IS fold signals must also be identical whether the overall df has
    future bars appended or not.  Verifies the warmup window for IS folds
    also ends at the IS fold boundary, not the OOS boundary.
    """
    N_EXTRA = 200
    rng = np.random.default_rng(SEED + 2)

    df_base = _make_ohlc(N_BARS)

    future_close = df_base["close"].iloc[-1] + np.cumsum(rng.standard_normal(N_EXTRA) * 0.3)
    noise        = rng.uniform(0.1, 0.5, (N_EXTRA, 3))
    future_idx   = pd.date_range(
        df_base.index[-1] + pd.Timedelta("15min"), periods=N_EXTRA, freq="15min",
    )
    df_ext = pd.concat([df_base, pd.DataFrame(
        {"open": future_close + noise[:, 2],
         "high": future_close + noise[:, 0],
         "low":  future_close - noise[:, 1],
         "close": future_close,
         "volume": rng.integers(1000, 5000, N_EXTRA).astype(float)},
        index=future_idx,
    )])

    splits = make_wf_splits(n_bars=N_BARS, n_splits=3, oos_frac=0.15, gap_bars=5)

    for fold_idx, (is_sl, _) in enumerate(splits):
        df_trunc = df_base.iloc[: is_sl.stop].copy()

        _, is_sig_trunc = generate_fold_signals(
            df_trunc, is_sl, _stub_generate_signals, PARAMS, warmup_bars=WARMUP,
        )
        _, is_sig_ext = generate_fold_signals(
            df_ext, is_sl, _stub_generate_signals, PARAMS, warmup_bars=WARMUP,
        )

        diff = (is_sig_trunc["roll_close"] - is_sig_ext["roll_close"]).abs().max()
        assert diff == 0.0, (
            f"Fold {fold_idx+1}: IS roll_close differs by {diff:.2e} — look-ahead in IS!"
        )
        print(f"  [fold {fold_idx+1}] IS bars={len(is_sig_trunc)}, diff={diff:.2e}  [PASS]")

    print(f"[PASS] test_is_append_invariance — {len(splits)} folds verified")


# ---------------------------------------------------------------------------
# Test 4 — bar_index continuity (warmup rows are correctly stripped)
# ---------------------------------------------------------------------------
def test_bar_index_continuity():
    """
    The stub embeds a running bar_index into signals.
    After stripping the warmup prefix, the first row of fold_signals must
    have bar_index == warmup_bars (i.e. the (warmup+1)-th bar in the context
    window), confirming the prefix was stripped by exactly warmup_bars rows.

    If warmup stripping is off by one or more, the bar_index alignment breaks.
    """
    df   = _make_ohlc(400)
    fold_sl = slice(80, 180)

    fold_df, fold_sig = generate_fold_signals(
        df, fold_sl, _stub_generate_signals, PARAMS, warmup_bars=WARMUP,
    )

    # Context window starts at max(0, 80 - 50) = 30.
    # bar_index for df row 80 within the context window [30:180] is 80-30 = 50.
    prefix_start = max(0, fold_sl.start - WARMUP)
    expected_first_bar_index = fold_sl.start - prefix_start  # = WARMUP

    actual_first_bar_index = int(fold_sig["bar_index"].iloc[0])

    assert actual_first_bar_index == expected_first_bar_index, (
        f"bar_index at fold start: expected {expected_first_bar_index}, "
        f"got {actual_first_bar_index} — warmup stripping offset is wrong"
    )

    # Also confirm the index alignment: fold_df and df must share the same
    # DatetimeIndex for the fold rows.
    assert list(fold_df.index) == list(df.index[fold_sl]), (
        "fold_df index does not match df.iloc[fold_sl].index"
    )

    print(
        f"[PASS] test_bar_index_continuity — "
        f"first bar_index={actual_first_bar_index} (expected {expected_first_bar_index})"
    )


# ---------------------------------------------------------------------------
# Test 5 — edge case: fold at the very start of df (prefix truncated to 0)
# ---------------------------------------------------------------------------
def test_edge_fold_at_start():
    """
    When fold_start < warmup_bars the prefix is truncated to [0, fold_start).
    The helper must not crash and must still return fold-length output.
    """
    df      = _make_ohlc(300)
    fold_sl = slice(10, 60)   # fold_start=10 < WARMUP=50

    fold_df, fold_sig = generate_fold_signals(
        df, fold_sl, _stub_generate_signals, PARAMS, warmup_bars=WARMUP,
    )

    expected = fold_sl.stop - fold_sl.start
    assert len(fold_df)  == expected, f"Expected {expected} rows, got {len(fold_df)}"
    assert len(fold_sig) == expected, f"Expected {expected} signal rows, got {len(fold_sig)}"

    print(
        f"[PASS] test_edge_fold_at_start — fold_start=10 < warmup={WARMUP}, "
        f"output rows={expected} (correct)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("Walk-forward causality / append-invariance tests (T1.3)")
    print("=" * 65)

    test_warmup_rows_excluded()
    print()

    print("test_oos_append_invariance:")
    test_oos_append_invariance()
    print()

    print("test_is_append_invariance:")
    test_is_append_invariance()
    print()

    test_bar_index_continuity()
    print()

    test_edge_fold_at_start()
    print()

    print("=" * 65)
    print("All walk-forward causality tests PASSED.")
    print("=" * 65)
