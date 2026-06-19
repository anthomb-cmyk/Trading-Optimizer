"""
tests/test_gap_fade_causality.py
================================
Causality (no-look-ahead) regression tests for the GapFade strategy.

Dependencies: numpy, pandas, ta only — no vectorbt / numba required.

What is verified
----------------
1. APPEND-INVARIANCE (the core no-repaint guarantee)
   Running generate_signals on df[:k] must produce EXACTLY the same signal
   columns (for every index < k - margin) as running it on df[:k+50].
   Appending 50 future bars must never change an already-emitted signal.
   This is the operational definition of strict causality for the whole
   strategy pipeline (session detection, gap, running extremes, entry,
   stop, target, confluence).

2. PURE TRUNCATION CAUSALITY
   For each bar t that fires a signal on the full frame, recomputing on the
   prefix df[:t+1] must still fire the same signal at t with the same entry,
   stop and target.  No bar after t may influence the decision at t.

3. SIGNALS ACTUALLY FIRE
   A causality test is vacuous if the strategy never signals.  We assert the
   synthetic frame produces a non-trivial number of triggers so the
   invariance checks above have teeth.

The synthetic frame is a seeded 5-minute OHLCV series spanning many days,
with an engineered gap injected at each New York RTH open so the gap-fade
edge is exercised.  Indicators (atr, ema_20/50/200, rsi, vol_ma, body,
range, bullish, swing cols) are computed inline so the test is
self-contained.
"""
import sys
import os

# Allow running directly with `python3 tests/test_gap_fade_causality.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend     import EMAIndicator
from ta.momentum  import RSIIndicator

from strategies.gap_fade import GapFade

SEED   = 7
N_DAYS = 30
FREQ   = "5min"

# Signal columns the strategy fills; these are what must be append-invariant.
_SIGNAL_COLS = [
    "long_signal", "short_signal", "entry_price",
    "stop_loss", "take_profit", "confluence_count", "valid",
]


# ---------------------------------------------------------------------------
# Synthetic OHLCV + session frame with engineered RTH opening gaps
# ---------------------------------------------------------------------------
def _make_frame(n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """
    Build a reproducible 5-minute OHLCV frame covering *n_days* calendar days
    of continuous 24h bars (so RTH sessions 13:30-22:00 UTC are surrounded by
    non-RTH bars and separated by overnight gaps).  At each RTH session open we
    inject a deterministic gap (alternating up/down) larger than the ATR so the
    fade edge fires.
    """
    rng = np.random.default_rng(seed)

    # Continuous 5-min index across n_days (24h/day) starting at a Monday.
    start = pd.Timestamp("2024-01-08 00:00", tz="UTC")  # Monday
    periods = n_days * 24 * 12  # 12 five-min bars per hour
    idx = pd.date_range(start, periods=periods, freq=FREQ)

    # Base random walk for close.
    steps = rng.standard_normal(periods) * 0.4
    close = 5000.0 + np.cumsum(steps)

    # Identify RTH open bars (first 13:30 UTC bar of each day) and inject a gap
    # into the close path so session_open - prior_rth_close is large.
    hour = idx.hour + idx.minute / 60.0
    is_rth = (hour >= 13.5) & (hour < 22.0)
    rth_arr = is_rth
    prev_rth = np.concatenate(([False], rth_arr[:-1]))
    rth_open_bar = rth_arr & (~prev_rth)

    # Apply a persistent jump at each RTH open (alternating sign, ~6 points
    # which is well above the typical ATR of this 0.4-std walk).
    gap_sign = 1
    jump = np.zeros(periods)
    for i in range(periods):
        if rth_open_bar[i]:
            jump[i] = gap_sign * 6.0
            gap_sign *= -1
    close = close + np.cumsum(jump)

    # Build OHLC around the (gapped) close path.
    noise_h = rng.uniform(0.2, 1.2, periods)
    noise_l = rng.uniform(0.2, 1.2, periods)
    high = close + noise_h
    low  = close - noise_l

    # open = previous close + small noise, EXCEPT at RTH open bars where we
    # force the open to reflect the gap (open near the new gapped close so the
    # session-open price genuinely differs from the prior RTH close).
    open_ = np.empty(periods)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.uniform(-0.2, 0.2, periods - 1)
    # At RTH open bars, set open = close (the gapped level) so sess_open carries
    # the jump relative to the prior session's final close.
    open_[rth_open_bar] = close[rth_open_bar] + rng.uniform(-0.1, 0.1, rth_open_bar.sum())

    # Enforce OHLC consistency.
    high = np.maximum(high, np.maximum(open_, close))
    low  = np.minimum(low,  np.minimum(open_, close))

    volume = rng.uniform(800, 2400, periods)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "datetime"
    return _add_indicators(df)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inline causal indicator layer mirroring data/loader.py._add_indicators so
    the test is self-contained.  All of these are causal (trailing windows /
    EMAs / pointwise) — none of them peek ahead.
    """
    df = df.copy()
    df["atr"]     = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr_avg"] = df["atr"].rolling(50).mean()

    for length in (20, 50, 200):
        df[f"ema_{length}"] = EMAIndicator(df["close"], window=length).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["vol_ma"] = df["volume"].rolling(20).mean()

    # Causal swing pivots (confirm-and-shift), lookback 10 like the loader.
    lb = 10
    df["swing_high"] = _swing_highs(df["high"].to_numpy(), lb)
    df["swing_low"]  = _swing_lows(df["low"].to_numpy(), lb)
    df["swing_high_price"] = df["high"].shift(lb).where(df["swing_high"])
    df["swing_low_price"]  = df["low"].shift(lb).where(df["swing_low"])

    df["body"]  = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["bullish"] = df["close"] > df["open"]
    return df


def _swing_highs(high: np.ndarray, lb: int) -> np.ndarray:
    """Causal swing-high confirm-and-shift (numpy only)."""
    n = len(high)
    out = np.zeros(n, dtype=bool)
    for t in range(2 * lb, n):
        c = t - lb
        window = high[t - 2 * lb : t + 1]
        if high[c] == window.max() and np.argmax(window) == lb:
            out[t] = True
    return out


def _swing_lows(low: np.ndarray, lb: int) -> np.ndarray:
    """Causal swing-low confirm-and-shift (numpy only)."""
    n = len(low)
    out = np.zeros(n, dtype=bool)
    for t in range(2 * lb, n):
        c = t - lb
        window = low[t - 2 * lb : t + 1]
        if low[c] == window.min() and np.argmin(window) == lb:
            out[t] = True
    return out


def _cols_equal(a: pd.Series, b: pd.Series) -> bool:
    """Element-wise equality treating NaN == NaN as equal."""
    av = a.to_numpy()
    bv = b.to_numpy()
    if av.shape != bv.shape:
        return False
    try:
        a_nan = pd.isna(av)
        b_nan = pd.isna(bv)
    except TypeError:
        a_nan = np.zeros(av.shape, dtype=bool)
        b_nan = np.zeros(bv.shape, dtype=bool)
    both_nan = a_nan & b_nan
    eq = np.zeros(av.shape, dtype=bool)
    non_nan = ~(a_nan | b_nan)
    eq[non_nan] = av[non_nan] == bv[non_nan]
    return bool(np.all(eq | both_nan))


# ---------------------------------------------------------------------------
# Test 1 — Append-invariance (no repaint)
# ---------------------------------------------------------------------------
def test_append_invariance():
    """
    Signal columns on df[:k] must equal those on df[:k+50] for every index
    < k - margin.  Appending future bars must not change any settled signal.
    """
    df = _make_frame()
    strat = GapFade(symbol="MES")
    params = strat.default_params

    margin = 2  # tiny guard band; the strategy emits at the entry bar (no fwd dep)
    n = len(df)
    cutpoints = [n // 3, n // 2, (2 * n) // 3]

    total_checked = 0
    for k in cutpoints:
        assert k + 50 <= n, "cutpoint + 50 must fit inside the frame"
        sig_short = strat.generate_signals(df.iloc[:k].copy(), params)
        sig_long  = strat.generate_signals(df.iloc[:k + 50].copy(), params)

        check_end = k - margin
        for col in _SIGNAL_COLS:
            s = sig_short[col].iloc[:check_end]
            l = sig_long[col].iloc[:check_end]
            assert _cols_equal(s, l), (
                f"append-invariance violated in column '{col}' at cutpoint k={k} "
                f"(first mismatch index "
                f"{int(np.argmax((s.to_numpy() != l.to_numpy()) & ~(pd.isna(s.to_numpy()) & pd.isna(l.to_numpy()))))})"
            )
        total_checked += check_end

    print(f"[PASS] test_append_invariance — {len(cutpoints)} cutpoints, "
          f"{total_checked} index-checks across {len(_SIGNAL_COLS)} columns")


# ---------------------------------------------------------------------------
# Test 2 — Pure truncation causality at every signalling bar
# ---------------------------------------------------------------------------
def test_truncation_causality():
    """
    For each bar t that fires a long/short signal on the full frame,
    recomputing on the prefix df[:t+1] must still fire that signal at t with
    the same entry / stop / target.  No future bar influences the decision.
    """
    df = _make_frame()
    strat = GapFade(symbol="MES")
    params = strat.default_params

    full = strat.generate_signals(df.copy(), params)
    fire = full["long_signal"].to_numpy() | full["short_signal"].to_numpy()
    fire_idx = np.where(fire)[0]

    assert len(fire_idx) > 0, "no signals fired — cannot validate truncation causality"

    checked = 0
    for t in fire_idx:
        trunc = strat.generate_signals(df.iloc[: t + 1].copy(), params)
        assert len(trunc) == t + 1
        for col in _SIGNAL_COLS:
            v_full  = full[col].iloc[t]
            v_trunc = trunc[col].iloc[t]
            both_nan = pd.isna(v_full) and pd.isna(v_trunc)
            assert both_nan or (v_full == v_trunc), (
                f"truncation causality violated at t={t}, column '{col}': "
                f"full={v_full}, truncated={v_trunc}"
            )
        checked += 1

    print(f"[PASS] test_truncation_causality — {checked} signalling bars verified "
          f"identical under prefix truncation")


# ---------------------------------------------------------------------------
# Test 3 — Signals actually fire (guards against a vacuous causality pass)
# ---------------------------------------------------------------------------
def test_signals_fire():
    """The engineered gaps must produce a meaningful number of fade signals."""
    df = _make_frame()
    strat = GapFade(symbol="MES")
    # Loosen min_confluence isn't needed; defaults should already fire.
    sig = strat.generate_signals(df.copy(), strat.default_params)
    n_long  = int(sig["long_signal"].sum())
    n_short = int(sig["short_signal"].sum())
    total = n_long + n_short

    assert total >= 5, (
        f"expected >= 5 gap-fade signals on the synthetic frame, got "
        f"{total} (long={n_long}, short={n_short}); causality test would be vacuous"
    )

    # Sanity: every fired signal must have finite entry/stop/target and the
    # geometry must be on the correct side (stop opposite target vs entry).
    fired = sig["long_signal"] | sig["short_signal"]
    f = sig[fired]
    assert f["entry_price"].notna().all(), "entry_price NaN on a fired signal"
    assert f["stop_loss"].notna().all(),   "stop_loss NaN on a fired signal"
    assert f["take_profit"].notna().all(), "take_profit NaN on a fired signal"

    longs = sig[sig["long_signal"]]
    shorts = sig[sig["short_signal"]]
    assert (longs["stop_loss"]   < longs["entry_price"]).all(),  "long stop must be below entry"
    assert (longs["take_profit"] > longs["entry_price"]).all(),  "long target must be above entry"
    assert (shorts["stop_loss"]   > shorts["entry_price"]).all(), "short stop must be above entry"
    assert (shorts["take_profit"] < shorts["entry_price"]).all(), "short target must be below entry"

    print(f"[PASS] test_signals_fire — {total} signals (long={n_long}, short={n_short}), "
          f"geometry valid")


# ---------------------------------------------------------------------------
# Entry point for both pytest and direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_signals_fire()
    test_append_invariance()
    test_truncation_causality()
    print("\nAll gap_fade causality tests passed.")
