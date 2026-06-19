"""
tests/test_short_term_reversal_causality.py
===========================================
Causality (no-look-ahead / append-invariance) tests for ShortTermReversal.

Dependencies: numpy, pandas, ta only — no vectorbt / numba required.

Guarantee verified
------------------
Append-invariance: the strategy's signal columns computed on a PREFIX
df[:k] must be IDENTICAL to the same columns computed on a LONGER frame
df[:k+50], for every index < k - margin.  Appending 50 future bars must
never change any signal, price, stop, target, or confluence value that the
strategy already emitted on a settled bar.  This is the operational
definition of "an output at bar t uses only data at indices <= t".

A second pure-causality check truncates the frame right after each fired
signal and confirms the signal still fires identically with no future bars.

The synthetic frame includes a DatetimeIndex (so the RTH session mask is
exercised), seeded OHLCV, and the indicator columns the strategy consumes
(atr, ema_20, rsi, atr_avg, vol_ma) computed inline via `ta`.
"""
import sys
import os

# Allow running directly with `python3 tests/test_short_term_reversal_causality.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

from strategies.short_term_reversal import ShortTermReversal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED   = 7
N_BARS = 600
# Columns the strategy writes that must be append-invariant on settled bars.
SIGNAL_COLS = [
    "long_signal", "short_signal", "entry_price",
    "stop_loss", "take_profit", "confluence_count", "valid",
]
# Margin: the longest backward window the strategy reads is the lookback (<=10),
# the RSI window (14), and the EMA-20.  EMAs are technically infinite-memory but
# converge; we therefore compare only the deterministic signal columns and use a
# generous margin so the comparison zone is unaffected by appended bars. Because
# the strategy only reads BACKWARD, settled bars are bit-identical regardless of
# margin — margin only trims the boundary region we bother to check.
MARGIN = 30


# ---------------------------------------------------------------------------
# Synthetic OHLCV + indicators
# ---------------------------------------------------------------------------
def _make_df(n: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """
    Seeded synthetic OHLCV with a DatetimeIndex and the indicator columns the
    strategy needs.  Injects occasional sharp moves so the reversal logic
    actually fires (otherwise a pure random walk rarely triggers).
    """
    rng   = np.random.default_rng(seed)
    steps = rng.standard_normal(n) * 0.5

    # Inject sharp impulse moves every ~37 bars to trigger the fade logic.
    for i in range(20, n, 37):
        steps[i] += rng.choice([-1.0, 1.0]) * rng.uniform(4.0, 7.0)

    close = 100.0 + np.cumsum(steps)

    noise_h = rng.uniform(0.1, 0.8, n)
    noise_l = rng.uniform(0.1, 0.8, n)
    noise_o = rng.uniform(-0.3, 0.3, n)

    high  = close + noise_h
    low   = close - noise_l
    open_ = close + noise_o
    high  = np.maximum(high, np.maximum(open_, close))
    low   = np.minimum(low,  np.minimum(open_, close))

    volume = rng.uniform(800, 1600, n)

    # Start at 13:30 UTC so a chunk of bars land inside the new_york RTH window.
    idx = pd.date_range("2024-03-04 13:30", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "datetime"
    return _add_indicators(df)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the causal indicator columns the strategy reads, inline via `ta`.
    Mirrors data/loader._add_indicators for the relevant columns only.
    NOTE: every column here is computed with backward-only windows, so they are
    themselves append-invariant on settled bars — exactly what lets the
    strategy be append-invariant.
    """
    df = df.copy()
    df["atr"]     = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr_avg"] = df["atr"].rolling(50).mean()
    df["ema_20"]  = EMAIndicator(df["close"], window=20).ema_indicator()
    df["rsi"]     = RSIIndicator(df["close"], window=14).rsi()
    df["vol_ma"]  = df["volume"].rolling(20).mean()
    return df


def _signals(df: pd.DataFrame) -> pd.DataFrame:
    strat = ShortTermReversal(symbol="MES")
    return strat.generate_signals(df, strat.default_params)


def _equal_cell(a, b) -> bool:
    """Treat NaN == NaN as equal; compare numerics/bools exactly otherwise."""
    a_nan = isinstance(a, float) and np.isnan(a)
    b_nan = isinstance(b, float) and np.isnan(b)
    if a_nan and b_nan:
        return True
    if a_nan != b_nan:
        return False
    return a == b


# ---------------------------------------------------------------------------
# Test 1 — Append-invariance: df[:k] vs df[:k+50]
# ---------------------------------------------------------------------------
def test_append_invariance():
    """
    Signal columns on df[:k] must equal those on df[:k+50] for all indices
    < k - MARGIN.  Adding 50 future bars must not repaint any settled signal.
    """
    df = _make_df()
    k  = 400
    assert k + 50 <= len(df)

    out_short = _signals(df.iloc[:k].copy())
    out_long  = _signals(df.iloc[:k + 50].copy())

    settled_end = k - MARGIN
    mismatches  = 0
    for col in SIGNAL_COLS:
        s = out_short[col].to_numpy()
        l = out_long[col].to_numpy()
        for t in range(settled_end):
            if not _equal_cell(s[t], l[t]):
                mismatches += 1
                raise AssertionError(
                    f"append-invariance violated in '{col}' at t={t}: "
                    f"short={s[t]!r}, long={l[t]!r}"
                )

    n_long  = int(out_short["long_signal"].iloc[:settled_end].sum())
    n_short = int(out_short["short_signal"].iloc[:settled_end].sum())
    assert (n_long + n_short) > 0, (
        "synthetic data produced NO signals — test would be vacuous; "
        "adjust the impulse injection."
    )
    print(
        f"[PASS] test_append_invariance — {settled_end} settled bars x "
        f"{len(SIGNAL_COLS)} cols checked, {mismatches} mismatches; "
        f"fired {n_long} long / {n_short} short in settled zone"
    )


# ---------------------------------------------------------------------------
# Test 2 — Pure causality: truncate right after each fired signal
# ---------------------------------------------------------------------------
def test_pure_causality_on_fires():
    """
    For every bar t where the full-frame strategy fires a long/short signal,
    recomputing on df[:t+1] (no future bars at all) must fire the IDENTICAL
    signal with the same entry/stop/target/confluence.  Equivalently, a False
    bar must stay False after truncation.
    """
    df  = _make_df()
    out = _signals(df)

    fired = out.index[(out["long_signal"] | out["short_signal"])]
    fired_pos = [out.index.get_loc(ix) for ix in fired]
    assert len(fired_pos) > 0, "no signals fired on full frame — test vacuous"

    checked = 0
    for t in fired_pos:
        trunc = _signals(df.iloc[: t + 1].copy())
        for col in SIGNAL_COLS:
            full_v  = out[col].iloc[t]
            trunc_v = trunc[col].iloc[t]
            assert _equal_cell(full_v, trunc_v), (
                f"pure-causality violated in '{col}' at fired bar t={t}: "
                f"full={full_v!r}, truncated={trunc_v!r}"
            )
        checked += 1

    # Spot-check a sample of NON-fired bars stay non-fired after truncation.
    non_fired = [t for t in range(MARGIN, len(df)) if t not in set(fired_pos)]
    for t in non_fired[::23]:
        trunc = _signals(df.iloc[: t + 1].copy())
        assert not bool(trunc["long_signal"].iloc[t]), f"spurious long at t={t}"
        assert not bool(trunc["short_signal"].iloc[t]), f"spurious short at t={t}"

    print(
        f"[PASS] test_pure_causality_on_fires — verified {checked} fired bars "
        f"are truncation-stable, plus sampled non-fired bars"
    )


# ---------------------------------------------------------------------------
# Test 3 — No negative shift / no future reference (structural smoke test)
# ---------------------------------------------------------------------------
def test_signal_uses_only_past_or_current():
    """
    Direct check: zero out everything strictly AFTER a fired bar t and confirm
    the signal at t is unchanged.  If the strategy peeked ahead, corrupting the
    future would change the past — it must not.
    """
    df  = _make_df()
    out = _signals(df)
    fired_pos = [out.index.get_loc(ix)
                 for ix in out.index[(out["long_signal"] | out["short_signal"])]]
    assert fired_pos, "no signals fired — test vacuous"

    # Pick a fired bar comfortably inside the series.
    t = next((p for p in fired_pos if MARGIN < p < len(df) - 50), fired_pos[0])

    corrupted = df.copy()
    # Replace all future OHLCV + indicators with garbage.
    fut = corrupted.index[t + 1:]
    rng = np.random.default_rng(123)
    for col in corrupted.columns:
        corrupted.loc[fut, col] = rng.uniform(1e4, 2e4, size=len(fut))

    out_corr = _signals(corrupted)
    for col in SIGNAL_COLS:
        assert _equal_cell(out[col].iloc[t], out_corr[col].iloc[t]), (
            f"future corruption changed '{col}' at past bar t={t} — look-ahead!"
        )

    print(f"[PASS] test_signal_uses_only_past_or_current — bar t={t} unaffected "
          f"by corrupted future")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_append_invariance()
    test_pure_causality_on_fires()
    test_signal_uses_only_past_or_current()
    print("\nAll ShortTermReversal causality tests passed.")
