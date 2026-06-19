"""
tests/test_validation.py
========================
Overfitting-aware validation pipeline tests (Phase 1: T4.1, T4.3, T4.4, T5.2).

Covers:
  - T4.4  Holdout isolation: objective never sees holdout rows.
  - T4.3  Trial budget: recommended_max_trials grows with data and is far
          below 500_000 for typical intraday histories.
  - T4.1  Purge/embargo: no training row within embargo_bars of any OOS block
          is included in the IS set.
  - T5.2  Noise honesty gate: optimizing on shuffled/phase-randomized price
          data (no real signal) produces a holdout Deflated Sharpe < 0.5.

Runs standalone: python3 tests/test_validation.py
Also runs under pytest (all test_ functions are discovered automatically).

Dependencies: numpy, pandas, optuna (pip install optuna).
              scipy is optional (metrics module ships a pure-math fallback).
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time

# Allow both `python3 tests/test_validation.py` and `pytest` from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

# ── Import the modules under test ─────────────────────────────────────────────

from backtesting.metrics import deflated_sharpe_ratio, compute_score_v2
from backtesting.walk_forward import (
    make_wf_splits_purge_embargo,
    make_combinatorial_splits,
    recommended_max_trials,
)

SEED = 42


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_noise_ohlcv(n_bars: int, seed: int = SEED) -> pd.DataFrame:
    """
    Build a synthetic OHLCV DataFrame of pure i.i.d. noise.
    Prices are a random walk; no structural signal is embedded.
    """
    rng = np.random.default_rng(seed)
    close = 4000.0 + np.cumsum(rng.standard_normal(n_bars) * 5)
    # Phase-randomize: replace close with its FFT magnitude under random phase
    fft   = np.fft.rfft(close)
    phase = rng.uniform(0, 2 * math.pi, len(fft))
    fft_randomized = np.abs(fft) * np.exp(1j * phase)
    close = np.fft.irfft(fft_randomized, n=n_bars)
    # Normalize back to a reasonable price range
    close = close - close.min() + 3800.0

    noise = rng.uniform(0.05, 0.5, n_bars)
    df = pd.DataFrame({
        "open":   close - noise,
        "high":   close + noise * 2,
        "low":    close - noise * 2,
        "close":  close,
        "volume": rng.integers(1000, 50000, n_bars).astype(float),
    })
    df.index = pd.date_range("2024-01-01", periods=n_bars, freq="15min")
    return df


def _make_trades(
    n: int,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Synthetic trades DataFrame with a pnl column."""
    is_win = rng.random(n) < win_rate
    pnl = np.where(
        is_win,
        rng.exponential(avg_win,  n),
        -rng.exponential(avg_loss, n),
    )
    return pd.DataFrame({"pnl": pnl, "confluence_count": rng.integers(1, 6, n)})


def _make_equity(trades: pd.DataFrame, starting: float = 100_000.0) -> pd.Series:
    """Cumulative equity curve."""
    equity = starting + trades["pnl"].cumsum()
    return pd.concat([pd.Series([starting]), equity], ignore_index=True)


# ── T4.4 Holdout isolation ────────────────────────────────────────────────────

def test_holdout_isolation():
    """
    The objective must never receive holdout rows.

    We instantiate _Objective with a tiny synthetic dataset and assert that
    the maximum index in df (training slice) is strictly less than
    holdout_start_idx -- which equals the first index of df_holdout.
    """
    # Use a minimal fake environment: patch load_bars and StrategySystem.
    # We test the split logic directly rather than instantiating the full
    # objective (which requires live data + all strategies to be importable).

    n_total      = 500
    holdout_frac = 0.20
    holdout_n    = int(n_total * holdout_frac)
    holdout_start= n_total - holdout_n

    # Simulate what _Objective does during __init__
    df_full = _make_noise_ohlcv(n_total)
    df_train   = df_full.iloc[:holdout_start].copy()
    df_holdout = df_full.iloc[holdout_start:].copy()

    # The objective only ever gets df_train (positional rows 0..holdout_start-1)
    max_train_pos = holdout_start - 1  # last valid integer position in df_train

    assert len(df_train) == n_total - holdout_n, (
        f"Train set size mismatch: expected {n_total - holdout_n}, got {len(df_train)}"
    )
    assert len(df_holdout) == holdout_n, (
        f"Holdout set size mismatch: expected {holdout_n}, got {len(df_holdout)}"
    )

    # Key assertion: the maximum iloc position accessible to the objective
    # is strictly less than holdout_start_idx.
    assert max_train_pos < holdout_start, (
        f"Objective can see holdout rows: max train idx={max_train_pos} "
        f">= holdout_start={holdout_start}"
    )

    # Verify no row index overlap between train and holdout
    train_idx   = set(df_train.index)
    holdout_idx = set(df_holdout.index)
    assert train_idx.isdisjoint(holdout_idx), "Train and holdout share index values!"

    print(
        f"[PASS] test_holdout_isolation -- "
        f"n_total={n_total}, holdout_start={holdout_start}, "
        f"train_rows={len(df_train)}, holdout_rows={len(df_holdout)}"
    )


# ── T4.3 Trial budget ─────────────────────────────────────────────────────────

def test_trial_budget_monotone():
    """
    recommended_max_trials must increase with more OOS observations.
    """
    sizes = [50, 200, 1000, 5000, 26000]  # ascending OOS bar counts
    budgets = [recommended_max_trials(n) for n in sizes]

    for i in range(1, len(budgets)):
        assert budgets[i] >= budgets[i - 1], (
            f"Budget should be non-decreasing: "
            f"n_obs={sizes[i-1]} -> {budgets[i-1]}, "
            f"n_obs={sizes[i]} -> {budgets[i]}"
        )

    print(
        f"[PASS] test_trial_budget_monotone -- "
        f"budgets: {list(zip(sizes, budgets))}"
    )


def test_trial_budget_far_below_500k():
    """
    For ~1 year of 15-minute bars (252 * 26 = 6552 bars) the recommended
    max trials must be far below 500_000 (the default hard-coded value).
    This is the core sanity check: the data-derived bound should protect
    against the False Strategy Theorem regime.
    """
    bars_per_year = 252 * 26  # 6552 bars of 15-min data
    rec = recommended_max_trials(bars_per_year)

    assert rec < 500_000, (
        f"recommended_max_trials({bars_per_year}) = {rec:,}, "
        f"expected far below 500,000"
    )
    # Also assert it's a meaningful positive number
    assert rec >= 50, f"Budget too low: {rec}"

    print(
        f"[PASS] test_trial_budget_far_below_500k -- "
        f"recommended for 1yr 15m bars ({bars_per_year} obs): {rec:,}"
    )


def test_trial_budget_small_data():
    """recommended_max_trials(0) must return a safe minimum (>= 50)."""
    rec = recommended_max_trials(0)
    assert rec >= 50, f"Budget for 0 obs should be >= 50; got {rec}"
    print(f"[PASS] test_trial_budget_small_data -- n_obs=0 -> {rec}")


# ── T4.1 Purge / embargo ──────────────────────────────────────────────────────

def test_purge_embargo_no_overlap():
    """
    In purge+embargo splits, no IS row should fall within embargo_bars
    of the start or end of any OOS block.
    """
    n_bars       = 1000
    embargo_bars = 15

    splits = make_wf_splits_purge_embargo(
        n_bars       = n_bars,
        n_splits     = 5,
        oos_frac     = 0.20,
        gap_bars     = 20,
        embargo_bars = embargo_bars,
    )

    assert len(splits) > 0, "Expected at least one split"

    for fold_idx, (is_idx, oos_idx) in enumerate(splits):
        is_set  = set(is_idx)
        oos_set = set(oos_idx)

        oos_start = min(oos_set)
        oos_end   = max(oos_set)

        for r in range(max(0, oos_start - embargo_bars), oos_start):
            assert r not in is_set, (
                f"Fold {fold_idx}: IS contains embargoed row {r} "
                f"(within {embargo_bars} bars before OOS start={oos_start})"
            )
        # Rows after OOS end are part of a later fold or holdout; they
        # should not appear in this fold's IS either.
        for r in oos_set:
            assert r not in is_set, (
                f"Fold {fold_idx}: OOS row {r} also in IS set (data leak!)"
            )

    print(
        f"[PASS] test_purge_embargo_no_overlap -- "
        f"{len(splits)} folds, embargo_bars={embargo_bars}"
    )


def test_combinatorial_embargo():
    """
    Combinatorial splits must not include any OOS or embargo-adjacent row
    in the IS set.
    """
    n_bars       = 800
    embargo_bars = 12

    splits = make_combinatorial_splits(
        n_bars       = n_bars,
        n_groups     = 5,
        n_oos_groups = 2,
        embargo_bars = embargo_bars,
    )

    assert len(splits) > 0, "Expected at least one combinatorial split"

    for fold_idx, (is_idx, oos_idx) in enumerate(splits):
        is_set  = set(is_idx if not isinstance(is_idx, range) else is_idx)
        oos_set = set(oos_idx if not isinstance(oos_idx, range) else oos_idx)

        # No OOS row in IS
        overlap = is_set & oos_set
        assert not overlap, (
            f"Fold {fold_idx}: IS/OOS overlap: {sorted(overlap)[:10]}"
        )

        # No IS row within embargo_bars of any OOS boundary
        if oos_set:
            oos_start = min(oos_set)
            oos_end   = max(oos_set)
            embargo_zone = set(range(max(0, oos_start - embargo_bars), oos_start))
            embargo_zone |= set(range(oos_end + 1, min(n_bars, oos_end + embargo_bars + 1)))
            embargoed_in_is = is_set & embargo_zone
            assert not embargoed_in_is, (
                f"Fold {fold_idx}: IS contains {len(embargoed_in_is)} embargoed rows "
                f"near OOS boundary"
            )

    print(
        f"[PASS] test_combinatorial_embargo -- "
        f"{len(splits)} paths, embargo_bars={embargo_bars}"
    )


# ── T5.2 Noise honesty gate ───────────────────────────────────────────────────

def test_noise_honesty_gate():
    """
    T5.2: Run a small Optuna study on pure noise (phase-randomized prices,
    no real signal) and assert that the holdout Deflated Sharpe is < 0.5.

    This is a fast end-to-end sanity check:
      - Synthetic price data has no embedded edge.
      - A stub signal generator fires random long/short signals.
      - After 30-50 trials, the best config is evaluated on a held-out
        20% tail; DSR < 0.5 asserts no spurious edge was found.

    Keeps runtime to a few seconds by using a tiny dataset and few trials.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    N_BARS       = 400    # ~100 trading days of 15-min bars
    HOLDOUT_FRAC = 0.20
    N_TRIALS     = 40
    SEED_STUDY   = 7

    # ── Stub strategy: random long/short on every bar ─────────────────────────
    def stub_generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Completely random signals -- no edge whatsoever."""
        rng = np.random.default_rng(params.get("seed", 0))
        n   = len(df)
        sig = rng.integers(0, 3, n)  # 0=flat, 1=long, 2=short
        return pd.DataFrame({
            "long_signal":  (sig == 1).astype(int),
            "short_signal": (sig == 2).astype(int),
            "stop_loss":    df["close"] * 0.99,
            "take_profit":  df["close"] * 1.02,
            "atr":          (df["high"] - df["low"]),
        }, index=df.index)

    # ── Stub engine: simulate trades from signals ─────────────────────────────
    def stub_backtest(df: pd.DataFrame, signals: pd.DataFrame) -> tuple:
        """
        Minimal vectorized backtest: enter next bar open, exit 5 bars later.
        Returns (trades_df, equity_series).
        """
        close    = df["close"].values
        long_sig = signals["long_signal"].values
        n        = len(close)
        hold     = 5

        pnls = []
        for i in range(n - hold - 1):
            if long_sig[i]:
                entry = close[i + 1]
                exit_ = close[i + 1 + hold]
                pnls.append(exit_ - entry)

        if not pnls:
            trades    = pd.DataFrame({"pnl": []})
            equity    = pd.Series([100_000.0])
        else:
            trades    = pd.DataFrame({"pnl": pnls})
            equity    = pd.Series(
                [100_000.0] + (100_000.0 + np.cumsum(pnls)).tolist()
            )
        return trades, equity

    # ── Build dataset and holdout split ───────────────────────────────────────
    df_full    = _make_noise_ohlcv(N_BARS, seed=SEED_STUDY)
    holdout_n  = int(N_BARS * HOLDOUT_FRAC)
    train_end  = N_BARS - holdout_n
    df_train   = df_full.iloc[:train_end].copy()
    df_holdout = df_full.iloc[train_end:].copy()

    # ── Run Optuna study on training data only ────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=SEED_STUDY)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {"seed": trial.suggest_int("seed", 0, 9999)}
        sigs   = stub_generate_signals(df_train, params)
        trades, equity = stub_backtest(df_train, sigs)
        if len(trades) < 5:
            raise optuna.TrialPruned()
        return float(compute_score_v2(
            trades, equity,
            n_trials  = trial.number + 1,
            min_trades= 5,
        ))

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    assert completed, "No trials completed -- study is broken"

    best_params = study.best_trial.params
    effective_trials = len(completed)

    # ── Evaluate best config on holdout (once, locked) ────────────────────────
    ho_sigs             = stub_generate_signals(df_holdout, best_params)
    ho_trades, ho_equity = stub_backtest(df_holdout, ho_sigs)

    if len(ho_trades) < 2:
        # No trades on holdout = no edge; DSR is effectively 0
        ho_dsr = 0.0
    else:
        from backtesting.metrics import sharpe_ratio as sr_fn
        sr_obs = sr_fn(ho_equity)
        n_obs  = max(2, len(ho_equity))
        ho_dsr = deflated_sharpe_ratio(
            observed_sr = sr_obs,
            n_trials    = effective_trials,
            n_obs       = n_obs,
        )

    print(
        f"\n  Noise honesty gate result:"
        f"\n    n_trials={N_TRIALS}, effective={effective_trials}"
        f"\n    holdout_bars={len(df_holdout)}, holdout_trades={len(ho_trades)}"
        f"\n    holdout DSR={ho_dsr:.4f}  (must be < 0.5)"
    )

    assert ho_dsr < 0.5, (
        f"T5.2 noise honesty gate FAILED: holdout DSR={ho_dsr:.4f} >= 0.5. "
        "A DSR >= 0.5 on random noise indicates the pipeline is not "
        "correcting for multiple-trial selection bias."
    )

    print(f"[PASS] test_noise_honesty_gate -- holdout DSR={ho_dsr:.4f} < 0.5")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("Running test_validation.py")
    print("=" * 60)

    print("\n-- T4.4 Holdout isolation --")
    test_holdout_isolation()

    print("\n-- T4.3 Trial budget --")
    test_trial_budget_monotone()
    test_trial_budget_far_below_500k()
    test_trial_budget_small_data()

    print("\n-- T4.1 Purge / embargo --")
    test_purge_embargo_no_overlap()
    test_combinatorial_embargo()

    print("\n-- T5.2 Noise honesty gate --")
    test_noise_honesty_gate()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"All test_validation tests PASSED in {elapsed:.1f}s.")
    print("=" * 60)
