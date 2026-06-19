"""
tests/test_opening_range_breakout_causality.py
==============================================
Causality (no look-ahead / no-repaint) regression tests for the
OpeningRangeBreakout strategy (strategies/opening_range_breakout.py).

Dependencies: numpy, pandas only — no vectorbt / numba / pytest / ta required
(ATR / EMA-200 / RSI / vol_ma are computed inline with pure pandas).

Run directly::

    python3 tests/test_opening_range_breakout_causality.py

Or via pytest (both modes are supported).

The key property (append-invariance)
------------------------------------
The opening-range breakout is session-anchored: the opening range, the
within-session bar position, the "first breakout per side per session" gate,
and the volume/EMA features must ALL be built from data UP TO the current bar
only.  We verify this with append-invariance:

    strategy.generate_signals(df[:k])  ==  strategy.generate_signals(df[:k+50])

for every index strictly before ``k - margin`` on every signal column.  Adding
future bars must not alter any already-emitted signal, entry, stop, target,
confluence count, or validity flag.

A second, stronger test truncates the frame at ``t+1`` for every bar that fired
a signal and confirms the signal is identical — i.e. the flag at ``t`` depends
solely on bars with index ``<= t``.
"""
import sys
import os

# Support direct execution from repo root OR from the tests/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from strategies.opening_range_breakout import OpeningRangeBreakout


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic OHLCV + session builder
# ─────────────────────────────────────────────────────────────────────────────

SEED   = 2027
# 15-minute bars across many UTC calendar days so the RTH (new_york) session is
# entered repeatedly and several opening ranges form.
N_BARS = 1400


def _make_df(n: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """
    Build a reproducible synthetic OHLCV DataFrame on a 15-minute UTC grid with
    every indicator column OpeningRangeBreakout consumes:

        open, high, low, close, volume, atr, ema_200, vol_ma

    The 15-minute spacing across ~14.5 calendar days means the New York RTH
    session (13:30–22:00 UTC) is entered on many days, producing multiple
    opening ranges and breakout windows so the test exercises real signals.

    Volume is given occasional spikes so the ``volume > vol_mult * vol_ma`` gate
    actually triggers; otherwise no breakout would ever confirm and the test
    would be vacuous.
    """
    rng = np.random.default_rng(seed)

    # 15-minute bars starting at a Monday 00:00 UTC.
    idx = pd.date_range("2024-03-04 00:00", periods=n, freq="15min", tz="UTC")

    # Price random walk with mild drift.
    returns = rng.standard_normal(n) * 0.35
    close   = 100.0 + np.cumsum(returns)

    body = rng.standard_normal(n) * 0.15
    open_ = close - body
    wick_up = np.abs(rng.standard_normal(n)) * 0.20
    wick_dn = np.abs(rng.standard_normal(n)) * 0.20
    high_ = np.maximum(open_, close) + wick_up
    low_  = np.minimum(open_, close) - wick_dn

    # Volume: baseline + intermittent spikes to exercise the volume gate.
    volume = rng.uniform(700.0, 1500.0, n)
    spikes = rng.random(n) < 0.18
    volume[spikes] *= rng.uniform(2.0, 4.0, spikes.sum())

    df = pd.DataFrame(
        {"open": open_, "high": high_, "low": low_, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "datetime"

    # ── Indicator columns (pure pandas, all causal/trailing) ──────────────────

    # ATR: 14-bar rolling mean of true range.
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"]     = tr.rolling(14, min_periods=1).mean()
    df["atr_avg"] = df["atr"].rolling(50, min_periods=1).mean()

    # EMA-200 (used by htf_bias). ewm is causal (trailing).
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI-14 (not strictly needed by ORB but cheap and mirrors loader columns).
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    # Volume moving average (trailing 20-bar).
    df["vol_ma"] = df["volume"].rolling(20, min_periods=1).mean()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Shared invariance helper
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLS = [
    "long_signal", "short_signal", "entry_price",
    "stop_loss", "take_profit", "confluence_count", "valid",
]

CUTPOINT   = 900   # k — split point well inside N_BARS
EXTRA_BARS = 50    # future bars appended
# Margin must clear the largest backward-looking window the strategy depends on.
# The dominant one here is the per-session opening-range / first-breakout logic,
# which is fully settled within the same session.  We use a generous margin so
# that index < CUTPOINT - MARGIN is guaranteed to sit in a session that closed
# before the cutpoint (the only session that could differ between df[:k] and
# df[:k+50] is the partial session containing/after k).  96 bars = one full UTC
# day at 15m, which comfortably brackets any single session.
MARGIN     = 110


def _check_invariant(name: str, short_s: pd.Series, long_s: pd.Series, settled_end: int) -> None:
    """Assert short_s[:settled_end] == long_s[:settled_end] element-wise."""
    for t in range(settled_end):
        v_short = short_s.iloc[t]
        v_long  = long_s.iloc[t]
        if pd.isna(v_short) and pd.isna(v_long):
            continue
        assert v_short == v_long, (
            f"[{name}] repaint at index {t}: short={v_short!r}, long={v_long!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — append-invariance across all signal columns and default params
# ─────────────────────────────────────────────────────────────────────────────

def test_orb_append_invariance_default():
    df = _make_df()
    k  = CUTPOINT
    settled_end = k - MARGIN

    strat  = OpeningRangeBreakout(symbol="MES")
    params = strat.default_params

    out_short = strat.generate_signals(df.iloc[:k].copy(),              params)
    out_long  = strat.generate_signals(df.iloc[:k + EXTRA_BARS].copy(), params)

    for col in SIGNAL_COLS:
        _check_invariant(f"default[{col}]", out_short[col], out_long[col], settled_end)

    n_long  = int(out_short["long_signal"].iloc[:settled_end].sum())
    n_short = int(out_short["short_signal"].iloc[:settled_end].sum())
    assert (n_long + n_short) > 0, (
        "No breakout signals fired in the settled zone — test would be vacuous. "
        "Adjust the synthetic data."
    )
    print(f"[PASS] test_orb_append_invariance_default — {settled_end} indices, "
          f"longs={n_long}, shorts={n_short}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — append-invariance under a non-default param set (tighter OR, low vol)
# ─────────────────────────────────────────────────────────────────────────────

def test_orb_append_invariance_alt_params():
    df = _make_df()
    k  = CUTPOINT
    settled_end = k - MARGIN

    strat  = OpeningRangeBreakout(symbol="MES")
    params = {
        "or_bars":      4,
        "vol_mult":     1.0,
        "atr_sl_mult":  0.0,       # stop exactly on opposite OR boundary
        "rr_ratio":     3.0,
        "session":      "new_york",
        "require_bias": True,      # exercise the hard-bias branch
    }

    out_short = strat.generate_signals(df.iloc[:k].copy(),              params)
    out_long  = strat.generate_signals(df.iloc[:k + EXTRA_BARS].copy(), params)

    for col in SIGNAL_COLS:
        _check_invariant(f"alt[{col}]", out_short[col], out_long[col], settled_end)

    print(f"[PASS] test_orb_append_invariance_alt_params — {settled_end} indices checked")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — pure causality: truncate at t+1 for every fired signal
# ─────────────────────────────────────────────────────────────────────────────

def test_orb_no_future_dependency():
    """
    For every bar t that fires a long or short signal on the full frame,
    recompute on df[:t+1] and assert the signal (and its entry/stop/target)
    is identical.  This proves the flag at t uses only data at indices <= t.
    """
    df    = _make_df()
    strat = OpeningRangeBreakout(symbol="MES")
    p     = strat.default_params

    out_full = strat.generate_signals(df.copy(), p)

    fired = np.where(
        out_full["long_signal"].values | out_full["short_signal"].values
    )[0]
    assert len(fired) > 0, "No signals on the full frame — cannot test causality."

    checked = 0
    for t in fired:
        sub = df.iloc[: t + 1].copy()
        out_sub = strat.generate_signals(sub, p)
        assert len(out_sub) == t + 1, f"length mismatch at t={t}"
        for col in SIGNAL_COLS:
            v_full = out_full[col].iloc[t]
            v_sub  = out_sub[col].iloc[t]
            if pd.isna(v_full) and pd.isna(v_sub):
                continue
            assert v_full == v_sub, (
                f"causality violated at t={t}, col={col}: "
                f"full={v_full!r}, truncated={v_sub!r}"
            )
        checked += 1

    print(f"[PASS] test_orb_no_future_dependency — verified {checked} fired signals")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — semantic guards (at most first breakout per side per session, geometry)
# ─────────────────────────────────────────────────────────────────────────────

def test_orb_first_per_session_and_geometry():
    """
    Sanity guards on the strategy semantics (not just causality):

      (a) At most ONE long and ONE short signal per RTH session.
      (b) Every long signal: entry > stop_loss and take_profit > entry.
          Every short signal: entry < stop_loss and take_profit < entry.
      (c) No signal fires outside the configured session.
    """
    df    = _make_df()
    strat = OpeningRangeBreakout(symbol="MES")
    p     = strat.default_params
    out   = strat.generate_signals(df.copy(), p)

    in_session = strat._session_mask(df, p["session"])
    sess_id    = strat._session_id(df, in_session)

    # (a) at most one per side per session
    longs  = out["long_signal"]
    shorts = out["short_signal"]
    per_sess_long  = longs.groupby(sess_id).sum()
    per_sess_short = shorts.groupby(sess_id).sum()
    assert (per_sess_long  <= 1).all(), "More than one long per session detected."
    assert (per_sess_short <= 1).all(), "More than one short per session detected."

    # (c) no signal outside the session
    assert not (longs & ~in_session).any(),  "Long signal fired outside session."
    assert not (shorts & ~in_session).any(), "Short signal fired outside session."

    # (b) geometry of entry/stop/target
    lmask = longs.values
    smask = shorts.values
    e = out["entry_price"].values
    s = out["stop_loss"].values
    tp = out["take_profit"].values

    if lmask.any():
        assert np.all(e[lmask] > s[lmask]),  "Long stop not below entry."
        assert np.all(tp[lmask] > e[lmask]), "Long take-profit not above entry."
    if smask.any():
        assert np.all(e[smask] < s[smask]),  "Short stop not above entry."
        assert np.all(tp[smask] < e[smask]), "Short take-profit not below entry."

    n_sess = int(sess_id.dropna().nunique())
    print(f"[PASS] test_orb_first_per_session_and_geometry — "
          f"{n_sess} sessions, longs={int(longs.sum())}, shorts={int(shorts.sum())}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — supports both pytest and direct execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running opening_range_breakout causality tests...\n")
    test_orb_append_invariance_default()
    test_orb_append_invariance_alt_params()
    test_orb_no_future_dependency()
    test_orb_first_per_session_and_geometry()
    print("\nAll opening_range_breakout causality tests passed.")
