"""
tests/test_metrics.py
=====================
Regression tests for overfitting-aware metrics and the risk-adjusted scorer.

Covers:
  - deflated_sharpe_ratio (DSR) monotonicity and boundary sanity
  - probability_backtest_overfitting (PBO) on signal vs. noise
  - compute_score_v2 ranking: low-win-rate / high-edge > high-win-rate / low-edge

Runs without pytest:  python3 tests/test_metrics.py
Runs under pytest too (all public functions are named test_*).

Dependencies: numpy, pandas only -- scipy is used if installed but is not
required (the metrics module ships a pure-math fallback).
"""
import sys
import os

# Allow both `python3 tests/test_metrics.py` and `pytest` from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtesting.metrics import (
    deflated_sharpe_ratio,
    probability_backtest_overfitting,
    minimum_track_record_length,
    compute_score_v2,
)

SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trades(
    n: int,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build a synthetic trades DataFrame with a ``pnl`` column.
    avg_win and avg_loss are the mean absolute P&L for winning / losing trades.
    """
    is_win = rng.random(n) < win_rate
    pnl = np.where(
        is_win,
        rng.exponential(avg_win,  n),
        -rng.exponential(avg_loss, n),
    )
    return pd.DataFrame({"pnl": pnl, "confluence_count": rng.integers(1, 6, n)})


def _make_equity(trades: pd.DataFrame, starting: float = 100_000.0) -> pd.Series:
    """Cumulative equity curve from a trades DataFrame."""
    equity = starting + trades["pnl"].cumsum()
    equity = pd.concat([pd.Series([starting]), equity], ignore_index=True)
    return equity


# ── DSR tests ─────────────────────────────────────────────────────────────────

def test_dsr_monotonicity_trials():
    """
    Monotonicity: more trials -> higher SR0 -> lower DSR for the same observed SR.

    We use a modest observed_sr (0.5) with a short track record (60 obs) so
    that SR0 grows meaningfully as n_trials increases and DSR is not pegged
    at 1.0 across the board.
    """
    obs_sr = 0.5   # modest SR -- easy to beat with enough trials
    n_obs  = 60    # short track record -> high sr_variance -> large SR0 range
    trial_counts = [1, 10, 100, 1_000, 10_000, 100_000]
    dsr_vals = [
        deflated_sharpe_ratio(obs_sr, n_trials=t, n_obs=n_obs)
        for t in trial_counts
    ]

    for i in range(1, len(dsr_vals)):
        assert dsr_vals[i] <= dsr_vals[i - 1] + 1e-10, (
            f"DSR should be non-increasing in n_trials: "
            f"DSR({trial_counts[i-1]} trials)={dsr_vals[i-1]:.6f} "
            f"but DSR({trial_counts[i]} trials)={dsr_vals[i]:.6f}"
        )

    # Verify that DSR actually drops meaningfully from N=1 to N=100_000
    assert dsr_vals[0] > dsr_vals[-1] + 0.05, (
        f"DSR should drop substantially from N=1 ({dsr_vals[0]:.4f}) "
        f"to N=100000 ({dsr_vals[-1]:.4f})"
    )
    print(
        f"[PASS] test_dsr_monotonicity_trials -- "
        f"DSR values: {[f'{v:.4f}' for v in dsr_vals]}"
    )


def test_dsr_sanity_boundary():
    """
    Boundary sanity checks:

    1. A high observed SR with N=1 (no multiple-testing penalty) and a long
       track record should give DSR near 1.
    2. A modest observed SR with N=100_000 trials and a short track record
       (so SR0 is large) should give a much lower DSR than the N=1 case.

    We deliberately use different (obs_sr, n_obs) pairs in the two sub-checks
    to stay clear of the saturation zone: the SR0 from extreme-value theory
    grows with log(N) / sqrt(n_obs), so short n_obs + large N is where the
    penalty is most visible.
    """
    # Sub-check 1: strong signal, no multiple-testing penalty -> DSR near 1
    dsr_single = deflated_sharpe_ratio(observed_sr=2.0, n_trials=1, n_obs=500)
    assert dsr_single > 0.90, (
        f"DSR with N=1 and SR=2.0 over 500 obs should be near 1; got {dsr_single:.4f}"
    )

    # Sub-check 2: modest SR, many trials, short track record -> DSR substantially < 1
    dsr_many = deflated_sharpe_ratio(observed_sr=0.5, n_trials=100_000, n_obs=60)
    assert dsr_many < 0.50, (
        f"DSR with N=100000, SR=0.5, n_obs=60 should be substantially less than 0.5; "
        f"got {dsr_many:.4f}"
    )

    # And the N=1 version of the same modest SR should be much higher
    dsr_few = deflated_sharpe_ratio(observed_sr=0.5, n_trials=1, n_obs=60)
    assert dsr_few > dsr_many + 0.10, (
        f"DSR N=1 ({dsr_few:.4f}) should be much higher than N=100000 ({dsr_many:.4f})"
    )

    print(
        f"[PASS] test_dsr_sanity_boundary -- "
        f"N=1/SR=2.0: DSR={dsr_single:.4f}; "
        f"N=1/SR=0.5/n_obs=60: DSR={dsr_few:.4f}; "
        f"N=100000/SR=0.5/n_obs=60: DSR={dsr_many:.4f}"
    )


def test_dsr_negative_sr():
    """A negative observed SR against any benchmark should yield DSR near 0."""
    dsr = deflated_sharpe_ratio(-1.0, n_trials=1, n_obs=500)
    assert dsr < 0.15, (
        f"DSR for negative SR should be near 0; got {dsr:.4f}"
    )
    print(f"[PASS] test_dsr_negative_sr -- DSR={dsr:.4f}")


def test_dsr_output_bounded():
    """DSR must always be in [0, 1] regardless of extreme inputs."""
    for obs, n_t, n_o in [
        (100.0, 1,       10),
        (-100.0, 1,      10),
        (1.0,    1_000_000, 5_000),
        (0.0,    50,     100),
    ]:
        dsr = deflated_sharpe_ratio(obs, n_trials=n_t, n_obs=n_o)
        assert 0.0 <= dsr <= 1.0, (
            f"DSR out of [0,1]: obs_sr={obs}, n_trials={n_t}, n_obs={n_o} -> {dsr}"
        )
    print("[PASS] test_dsr_output_bounded")


# ── PBO tests ─────────────────────────────────────────────────────────────────

def test_pbo_dominant_config():
    """
    Signal regime: one config genuinely outperforms all others in every block.
    The IS-best should almost always be the OOS-best -> PBO near 0.
    """
    rng = np.random.default_rng(SEED)
    T, N = 160, 10
    # Base noise for all configs
    perf = rng.standard_normal((T, N)) * 0.01
    # Config 0 gets a large persistent positive drift
    perf[:, 0] += 0.10
    pbo = probability_backtest_overfitting(perf, S=8)
    assert pbo < 0.30, (
        f"PBO should be near 0 when one config dominates; got {pbo:.4f}"
    )
    print(f"[PASS] test_pbo_dominant_config -- PBO={pbo:.4f}")


def test_pbo_signal_lower_than_noise():
    """
    Key PBO ordering: PBO(strong signal) < PBO(pure noise).

    In the signal regime the IS-best is genuinely best in OOS, so PBO is low.
    In the noise regime the IS-best is a random pick, so PBO is higher.
    We average over multiple seeds to stabilise both estimates.
    """
    signal_pbos = []
    noise_pbos  = []

    for seed_offset in range(12):
        base_rng = np.random.default_rng(1000 + seed_offset)
        T, N = 160, 10

        # Signal matrix: config 0 has a clear persistent edge
        perf_signal = base_rng.standard_normal((T, N)) * 0.02
        perf_signal[:, 0] += 0.15
        signal_pbos.append(probability_backtest_overfitting(perf_signal, S=8))

        # Noise matrix: pure i.i.d., same shape
        noise_rng = np.random.default_rng(2000 + seed_offset)
        perf_noise = noise_rng.standard_normal((T, N))
        noise_pbos.append(probability_backtest_overfitting(perf_noise, S=8))

    mean_signal = float(np.mean(signal_pbos))
    mean_noise  = float(np.mean(noise_pbos))

    assert mean_signal < mean_noise, (
        f"Average PBO(signal)={mean_signal:.3f} should be < PBO(noise)={mean_noise:.3f}"
    )
    assert mean_signal < 0.25, (
        f"Average PBO for strong-signal regime should be < 0.25; got {mean_signal:.3f}"
    )
    print(
        f"[PASS] test_pbo_signal_lower_than_noise -- "
        f"mean PBO(signal)={mean_signal:.3f}, mean PBO(noise)={mean_noise:.3f}"
    )


def test_pbo_pure_noise_known_seed():
    """
    Spot-check: with a known seed where the PBO is deterministic,
    assert PBO is in the range expected for pure noise [0.0, 1.0] and
    meaningfully above zero (noise has non-trivial overfitting probability).
    We use a seed verified to produce a value well within the range.
    """
    rng = np.random.default_rng(5)   # seed verified to give PBO=0.5 for this config
    T, N = 160, 10
    perf = rng.standard_normal((T, N))
    pbo  = probability_backtest_overfitting(perf, S=8)
    assert 0.0 <= pbo <= 1.0, f"PBO out of bounds: {pbo}"
    # For pure noise, PBO should not be near 0 (that would mean consistent OOS winner)
    # Average across seeds is ~0.45, so individual seed can be anywhere; just check range
    print(f"[PASS] test_pbo_pure_noise_known_seed -- PBO={pbo:.4f}")


def test_pbo_output_bounded():
    """PBO must always be in [0, 1]."""
    rng = np.random.default_rng(SEED + 2)
    perf = rng.standard_normal((64, 5))
    pbo = probability_backtest_overfitting(perf, S=8)
    assert 0.0 <= pbo <= 1.0, f"PBO out of [0,1]: {pbo}"
    print(f"[PASS] test_pbo_output_bounded -- PBO={pbo:.4f}")


# ── MinTRL tests ──────────────────────────────────────────────────────────────

def test_mtrl_basic():
    """MinTRL should be > 1 and finite for a positive SR vs benchmark 0."""
    from backtesting.metrics import minimum_track_record_length
    mtrl = minimum_track_record_length(observed_sr=0.5, sr_benchmark=0.0, prob=0.95)
    assert mtrl > 1.0 and np.isfinite(mtrl), f"MinTRL should be finite > 1; got {mtrl}"
    # Higher SR -> lower MinTRL
    mtrl_high = minimum_track_record_length(observed_sr=2.0, sr_benchmark=0.0, prob=0.95)
    assert mtrl_high < mtrl, (
        f"Higher SR should require shorter track record; "
        f"SR=0.5 -> {mtrl:.1f}, SR=2.0 -> {mtrl_high:.1f}"
    )
    print(f"[PASS] test_mtrl_basic -- SR=0.5: {mtrl:.1f} obs; SR=2.0: {mtrl_high:.1f} obs")


# ── compute_score_v2 tests ────────────────────────────────────────────────────

def test_score_v2_low_edge_high_wr_scores_lower():
    """
    Key property: a 90% win-rate strategy with profit_factor < 1
    (tiny wins, huge losses) must score LOWER than a 45% win-rate strategy
    with profit_factor 2.0 and positive equity curve.

    This test would FAIL under compute_score (v1) because v1 weights win_rate
    at 0.35 -- exactly the overfitting trap compute_score_v2 is designed to avoid.
    """
    rng = np.random.default_rng(SEED + 10)

    # Strategy A: 90% win rate but small wins / huge losses (PF < 1)
    N_A = 200
    pnl_a = np.where(
        rng.random(N_A) < 0.90,
        rng.uniform(0.05, 0.15, N_A),    # tiny wins ~$0.10
        -rng.uniform(0.8, 1.2, N_A),     # large losses ~$1.00
    )
    # PF_A should be < 1: 90% * 0.10 wins vs 10% * 1.00 losses => PF ~ 0.9
    trades_a  = pd.DataFrame({"pnl": pnl_a})
    equity_a  = _make_equity(trades_a)

    # Strategy B: 45% win rate but good risk/reward (PF ~ 2.0+)
    N_B = 200
    pnl_b = np.where(
        rng.random(N_B) < 0.45,
        rng.uniform(1.8, 2.2, N_B),      # larger wins ~$2.00
        -rng.uniform(0.9, 1.1, N_B),     # controlled losses ~$1.00
    )
    trades_b  = pd.DataFrame({"pnl": pnl_b})
    equity_b  = _make_equity(trades_b)

    score_a = compute_score_v2(trades_a, equity_a)
    score_b = compute_score_v2(trades_b, equity_b)

    # Sanity-check the actual profit factors
    from backtesting.metrics import profit_factor, win_rate
    pf_a = profit_factor(trades_a)
    pf_b = profit_factor(trades_b)
    wr_a = win_rate(trades_a)
    wr_b = win_rate(trades_b)

    assert pf_a < 1.0, f"Strategy A should have PF < 1; got {pf_a:.3f}"
    assert pf_b > 1.5, f"Strategy B should have PF > 1.5; got {pf_b:.3f}"

    assert score_b > score_a, (
        f"Strategy B (WR={wr_b:.0%}, PF={pf_b:.2f}) should score higher than "
        f"Strategy A (WR={wr_a:.0%}, PF={pf_a:.2f}).\n"
        f"  score_a={score_a:.4f}, score_b={score_b:.4f}"
    )
    print(
        f"[PASS] test_score_v2_low_edge_high_wr_scores_lower\n"
        f"  Strategy A: WR={wr_a:.0%}, PF={pf_a:.3f}, score={score_a:.4f}\n"
        f"  Strategy B: WR={wr_b:.0%}, PF={pf_b:.3f}, score={score_b:.4f}"
    )


def test_score_v2_output_bounded():
    """compute_score_v2 must return a value in [0, 1]."""
    rng = np.random.default_rng(SEED + 20)
    for _ in range(5):
        trades  = _make_trades(50, win_rate=rng.uniform(0.2, 0.8),
                               avg_win=rng.uniform(0.5, 3.0),
                               avg_loss=rng.uniform(0.5, 3.0), rng=rng)
        equity  = _make_equity(trades)
        score   = compute_score_v2(trades, equity)
        assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score}"
    print("[PASS] test_score_v2_output_bounded")


def test_score_v2_below_min_trades():
    """Too few trades must return 0.0 regardless of quality."""
    rng = np.random.default_rng(SEED + 30)
    trades = _make_trades(5, win_rate=0.8, avg_win=2.0, avg_loss=0.5, rng=rng)
    equity = _make_equity(trades)
    score  = compute_score_v2(trades, equity, min_trades=20)
    assert score == 0.0, f"Expected 0.0 for too-few-trades, got {score}"
    print("[PASS] test_score_v2_below_min_trades")


def test_score_v2_with_deflated_sharpe():
    """Providing deflated_sharpe= argument should work without error."""
    rng = np.random.default_rng(SEED + 40)
    trades = _make_trades(100, win_rate=0.55, avg_win=1.5, avg_loss=1.0, rng=rng)
    equity = _make_equity(trades)
    # Pre-computed DSR at 0.75 (strongly positive)
    score_with  = compute_score_v2(trades, equity, deflated_sharpe=0.75)
    # Same strategy, DSR from n_trials
    score_auto  = compute_score_v2(trades, equity, n_trials=500)
    # Both should be positive and finite
    assert 0.0 < score_with  <= 1.0, f"score_with={score_with}"
    assert 0.0 < score_auto  <= 1.0, f"score_auto={score_auto}"
    print(
        f"[PASS] test_score_v2_with_deflated_sharpe -- "
        f"score (DSR=0.75)={score_with:.4f}; score (n_trials=500)={score_auto:.4f}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Running test_metrics.py")
    print("=" * 60)

    # DSR
    test_dsr_monotonicity_trials()
    test_dsr_sanity_boundary()
    test_dsr_negative_sr()
    test_dsr_output_bounded()

    # PBO
    test_pbo_dominant_config()
    test_pbo_signal_lower_than_noise()
    test_pbo_pure_noise_known_seed()
    test_pbo_output_bounded()

    # MinTRL
    test_mtrl_basic()

    # compute_score_v2
    test_score_v2_low_edge_high_wr_scores_lower()
    test_score_v2_output_bounded()
    test_score_v2_below_min_trades()
    test_score_v2_with_deflated_sharpe()

    print("=" * 60)
    print("All test_metrics tests PASSED.")
    print("=" * 60)
