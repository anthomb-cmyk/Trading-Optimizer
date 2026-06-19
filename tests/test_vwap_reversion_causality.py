"""
tests/test_vwap_reversion_causality.py
======================================
Causality (no-look-ahead / append-invariance) tests for the
VWAPReversion strategy.

Dependencies: numpy, pandas, ta only — no vectorbt / numba required.

What is verified
----------------
1. Append-invariance of the signal columns
   The strategy's signal columns computed on df[:k] must EXACTLY equal the
   same columns computed on df[:k+50], for every index < k - MARGIN.
   Extending the frame with 50 future bars must never change an already-emitted
   signal, entry, stop, target, confluence, or validity flag.  This is the
   operational definition of "an output at bar t uses only data at indices <= t".

2. Session VWAP reset + causality
   The session-anchored VWAP must (a) reset at each RTH session boundary and
   (b) be append-invariant inside each session.  We re-derive it manually for a
   single session and confirm it is a pure backward cumulative ratio.

The synthetic frame deliberately spans several days so multiple New York RTH
sessions (13:30–22:00 UTC) are present and the VWAP reset path is exercised.
"""
import sys
import os

# Allow running directly with `python3 tests/test_vwap_reversion_causality.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from ta.volatility import AverageTrueRange
from ta.trend     import EMAIndicator
from ta.momentum  import RSIIndicator

from strategies.vwap_reversion import VWAPReversion


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED   = 7
N_BARS = 1200          # ~12.5 trading days of 15m bars (96 bars/day)
MARGIN = 60            # settle band: ATR(14)/EMA(200)/RSI(14)/disp(20) warmup
SIGNAL_COLS = [
    "long_signal", "short_signal",
    "entry_price", "stop_loss", "take_profit",
    "confluence_count", "valid",
]


# ---------------------------------------------------------------------------
# Synthetic OHLCV + indicator frame
# ---------------------------------------------------------------------------
def _make_frame(n: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """
    Build a reproducible synthetic 15m OHLCV DataFrame with the indicator
    columns the strategy consumes (atr, atr_avg, ema_*, rsi, vol_ma, body,
    range, bullish, ...), computed exactly the way data/loader._add_indicators
    does.  A mild mean-reverting wobble around a slow drift is injected so the
    VWAP-reversion logic actually fires.
    """
    rng = np.random.default_rng(seed)

    drift = np.cumsum(rng.standard_normal(n) * 0.3)
    wobble = 4.0 * np.sin(np.arange(n) * 2 * np.pi / 48.0)   # intraday swing
    close = 100.0 + drift + wobble

    noise_h = rng.uniform(0.1, 1.2, n)
    noise_l = rng.uniform(0.1, 1.2, n)
    noise_o = rng.uniform(-0.6, 0.6, n)

    open_ = close + noise_o
    high  = np.maximum(close, open_) + noise_h
    low   = np.minimum(close, open_) - noise_l
    volume = rng.uniform(800, 5000, n).round()

    # 15-minute bars starting at a fixed UTC timestamp; spans many NY sessions.
    idx = pd.date_range("2024-03-04 00:00", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "datetime"

    return _add_indicators(df)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror data/loader._add_indicators for the columns the strategy reads."""
    df = df.copy()
    df["atr"]     = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr_avg"] = df["atr"].rolling(50).mean()

    for length in (20, 50, 200):
        df[f"ema_{length}"] = EMAIndicator(df["close"], window=length).ema_indicator()

    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

    df["vol_ma"] = df["volume"].rolling(20).mean()

    df["body"]       = (df["close"] - df["open"]).abs()
    df["range"]      = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["bullish"]    = df["close"] > df["open"]
    return df


def _equal_series(a: pd.Series, b: pd.Series, atol: float = 0.0) -> bool:
    """
    Element-wise equality treating NaN == NaN as equal (object/float safe).

    atol=0.0 (default) demands bit-identical values — used for append-invariance,
    where the SAME computation on overlapping prefixes must agree exactly.
    A positive atol is used only when comparing two DIFFERENT computation methods
    of the same quantity (vectorized vs. an explicit Python accumulation), where
    floating-point accumulation order can differ in the last ULP.
    """
    av = a.to_numpy()
    bv = b.to_numpy()
    if av.shape != bv.shape:
        return False
    out = np.empty(len(av), dtype=bool)
    for i in range(len(av)):
        x, y = av[i], bv[i]
        x_nan = isinstance(x, float) and np.isnan(x)
        y_nan = isinstance(y, float) and np.isnan(y)
        if x_nan and y_nan:
            out[i] = True
        elif x_nan != y_nan:
            out[i] = False
        elif atol > 0.0 and isinstance(x, float) and isinstance(y, float):
            out[i] = abs(x - y) <= atol
        else:
            out[i] = (x == y)
    return bool(out.all())


# ---------------------------------------------------------------------------
# Test 1 — Append-invariance of all signal columns
# ---------------------------------------------------------------------------
def test_signal_append_invariance():
    """
    For multiple cut points k, the strategy's signal columns on df[:k] must
    equal those on df[:k+50] for every index < k - MARGIN.
    """
    df    = _make_frame()
    strat = VWAPReversion(symbol="MES")
    params = strat.default_params

    cut_points = [300, 500, 800, 1000]
    total_long = total_short = 0

    for k in cut_points:
        short = strat.generate_signals(df.iloc[:k].copy(),      params)
        long_ = strat.generate_signals(df.iloc[:k + 50].copy(), params)

        settled_end = k - MARGIN
        assert settled_end > MARGIN, "cut point too small for the settle band"

        for col in SIGNAL_COLS:
            s = short[col].iloc[:settled_end]
            l = long_[col].iloc[:settled_end]
            assert _equal_series(s, l), (
                f"append-invariance violated for column '{col}' at cut k={k}: "
                f"a future-bar extension changed a settled value before index {settled_end}"
            )

        total_long  += int(short["long_signal"].iloc[:settled_end].sum())
        total_short += int(short["short_signal"].iloc[:settled_end].sum())

    print(f"[PASS] test_signal_append_invariance — {len(cut_points)} cut points, "
          f"settled longs={total_long}, settled shorts={total_short}, "
          f"all {len(SIGNAL_COLS)} signal columns invariant under +50 future bars")


# ---------------------------------------------------------------------------
# Test 2 — Strict truncation causality of the signal columns
# ---------------------------------------------------------------------------
def test_signal_truncation_causality():
    """
    A stronger, pure form: the signal at bar t computed on the FULL frame must
    equal the signal at bar t computed on the frame truncated at t+1, i.e. with
    every future bar removed.  Checked at all signal bars plus a sample of bars.

    Note: indicator columns (atr/rsi/ema) are recomputed from the truncated raw
    OHLCV so the recomputation itself must be causal too; truncating at t+1 makes
    bar t the last bar, leaving no opportunity for look-ahead.
    """
    df    = _make_frame()
    strat = VWAPReversion(symbol="MES")
    params = strat.default_params

    raw = df[["open", "high", "low", "close", "volume"]]
    full = strat.generate_signals(df.copy(), params)

    sig_bars = np.where(
        (full["long_signal"] | full["short_signal"]).to_numpy()
    )[0]
    # Only test bars past the warmup band so indicators are defined.
    sig_bars = [int(t) for t in sig_bars if t >= MARGIN]

    # Plus a sample of non-signal bars for spurious-True protection.
    sample = list(range(MARGIN, len(df), 97))
    check_bars = sorted(set(sig_bars) | set(sample))

    checked = 0
    for t in check_bars:
        trunc_raw = raw.iloc[: t + 1].copy()
        trunc_df  = _add_indicators(trunc_raw)
        trunc_sig = strat.generate_signals(trunc_df, params)
        assert len(trunc_sig) == t + 1

        for col in ("long_signal", "short_signal"):
            assert bool(trunc_sig[col].iloc[t]) == bool(full[col].iloc[t]), (
                f"truncation causality violated for '{col}' at t={t}: "
                f"full={bool(full[col].iloc[t])}, truncated={bool(trunc_sig[col].iloc[t])}"
            )
        checked += 1

    print(f"[PASS] test_signal_truncation_causality — {len(sig_bars)} signal bars + "
          f"{len(sample)} sampled bars ({checked} total) match under hard truncation")


# ---------------------------------------------------------------------------
# Test 3 — Session VWAP: causal cumulative ratio that resets each session
# ---------------------------------------------------------------------------
def test_session_vwap_reset_and_causality():
    """
    Verify the session-anchored VWAP:
      (a) resets at each New York session boundary (the first in-session bar's
          VWAP equals that bar's typical price);
      (b) equals the manual backward cumulative ratio over the running session;
      (c) is append-invariant: VWAP on df[:k] == VWAP on df[:k+50] for settled
          indices.
    """
    df = _make_frame()
    in_sess = VWAPReversion._session_mask(df, "new_york")
    vwap = VWAPReversion._session_vwap(df, in_sess)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol     = df["volume"].astype(float)

    # (a) + (b): manual per-session cumulative ratio, reset on each session start.
    sid = VWAPReversion._session_id(in_sess)
    manual = pd.Series(np.nan, index=df.index)
    for _id, grp_idx in pd.Series(range(len(df)), index=df.index).groupby(sid).groups.items():
        pos = [df.index.get_loc(ix) for ix in grp_idx]
        in_mask = in_sess.to_numpy()[pos]
        cum_pv = 0.0
        cum_vv = 0.0
        for j, abspos in enumerate(pos):
            if in_mask[j]:
                cum_pv += typical.iloc[abspos] * vol.iloc[abspos]
                cum_vv += vol.iloc[abspos]
                manual.iloc[abspos] = cum_pv / cum_vv if cum_vv > 0 else np.nan

    # Cross-method comparison (vectorized groupby cumsum vs. explicit Python
    # accumulation): allow last-ULP float drift, NaN pattern must match exactly.
    assert _equal_series(vwap, manual, atol=1e-9), (
        "session VWAP does not match the manual per-session backward cumulative ratio"
    )

    # First in-session bar of each session: VWAP must equal that bar's typical.
    first_bars = np.where((in_sess & ~in_sess.shift(1, fill_value=False)).to_numpy())[0]
    n_sessions = 0
    for t in first_bars:
        v = vwap.iloc[t]
        if np.isnan(v):
            continue
        assert abs(v - typical.iloc[t]) < 1e-9, (
            f"session VWAP did not reset at session-start bar t={t}: "
            f"vwap={v}, typical={typical.iloc[t]}"
        )
        n_sessions += 1
    assert n_sessions >= 2, "synthetic frame should contain multiple NY sessions"

    # (c) Append-invariance of the VWAP itself.
    k = 700
    settled_end = k - MARGIN
    in_s_short = VWAPReversion._session_mask(df.iloc[:k], "new_york")
    in_s_long  = VWAPReversion._session_mask(df.iloc[:k + 50], "new_york")
    vw_short = VWAPReversion._session_vwap(df.iloc[:k].copy(),      in_s_short)
    vw_long  = VWAPReversion._session_vwap(df.iloc[:k + 50].copy(), in_s_long)
    assert _equal_series(vw_short.iloc[:settled_end], vw_long.iloc[:settled_end]), (
        "session VWAP is not append-invariant under +50 future bars"
    )

    print(f"[PASS] test_session_vwap_reset_and_causality — {n_sessions} NY sessions reset "
          f"correctly, VWAP == manual cumulative ratio, append-invariant over "
          f"{settled_end} indices")


# ---------------------------------------------------------------------------
# Entry point for both pytest and direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_signal_append_invariance()
    test_signal_truncation_causality()
    test_session_vwap_reset_and_causality()
    print("\nAll VWAP-reversion causality tests passed.")
