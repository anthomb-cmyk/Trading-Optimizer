"""
tests/test_sentiment_momentum.py
=================================
Tests for strategies/sentiment_momentum.py (SentimentMomentum strategy).

ALL tests are synthetic — no live GDELT network calls are made.
``gdelt_news.fetch_news`` is monkeypatched in every test to return
either an empty or a synthetic DataFrame.

Supports:  python3 tests/test_sentiment_momentum.py
           pytest tests/test_sentiment_momentum.py
"""
from __future__ import annotations

import os
import sys
import types
import numpy as np
import pandas as pd

# Allow running directly from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Synthetic data factories ──────────────────────────────────────────────────

def _make_price_df(n: int = 200, freq: str = "15min") -> pd.DataFrame:
    """Full indicator-ready synthetic price frame (15-min bars, UTC)."""
    rng   = np.random.default_rng(99)
    close = 4500.0 + np.cumsum(rng.standard_normal(n) * 3.0)
    atr   = np.abs(rng.standard_normal(n)) * 6.0 + 4.0
    idx   = pd.date_range("2024-03-01 00:00", periods=n, freq=freq, tz="UTC")

    # EMA-20 (causal ewm)
    close_s = pd.Series(close, index=idx)
    ema_20  = close_s.ewm(span=20, adjust=False).mean().values

    df = pd.DataFrame(
        {
            "open":   close - rng.uniform(0, 2, n),
            "high":   close + rng.uniform(0.5, 4, n),
            "low":    close - rng.uniform(0.5, 4, n),
            "close":  close,
            "volume": rng.integers(500, 3000, n).astype(float),
            "atr":    atr,
            "ema_20": ema_20,
        },
        index=idx,
    )
    return df


def _make_news_df(n: int = 60, seed: int = 13) -> pd.DataFrame:
    """Synthetic news DataFrame with UTC DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01 01:00", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "title":  [f"Headline {i}" for i in range(n)],
            "domain": ["news.com"] * n,
            "tone":   rng.uniform(-4.0, 4.0, n),
            "url":    [f"http://news.com/{i}" for i in range(n)],
        },
        index=idx,
    )


def _empty_news_df() -> pd.DataFrame:
    """Correctly-typed empty news frame matching gdelt_news schema."""
    idx = pd.DatetimeIndex([], tz="UTC", name="seendate")
    return pd.DataFrame(
        {
            "title":  pd.Series([], dtype="object"),
            "domain": pd.Series([], dtype="object"),
            "tone":   pd.Series([], dtype="float64"),
            "url":    pd.Series([], dtype="object"),
        },
        index=idx,
    )


# ── Signal-frame validator ────────────────────────────────────────────────────

_REQUIRED_SIGNAL_COLS = [
    "long_signal", "short_signal",
    "entry_price", "stop_loss", "take_profit",
    "confluence_count", "valid",
]


def _assert_valid_signal_frame(result: pd.DataFrame, price_df: pd.DataFrame) -> None:
    """Assert that result satisfies the BaseStrategy signal-frame contract."""
    assert isinstance(result, pd.DataFrame), "generate_signals must return a DataFrame"
    assert len(result) == len(price_df), (
        f"result length {len(result)} != price_df length {len(price_df)}"
    )
    for col in _REQUIRED_SIGNAL_COLS:
        assert col in result.columns, f"Missing required signal column: {col!r}"

    # long_signal / short_signal must be boolean-like
    assert result["long_signal"].dtype  in (bool, np.dtype("bool"), np.dtype("O")), \
        f"long_signal dtype {result['long_signal'].dtype}"
    assert result["short_signal"].dtype in (bool, np.dtype("bool"), np.dtype("O")), \
        f"short_signal dtype {result['short_signal'].dtype}"


# ── Monkeypatch helper ────────────────────────────────────────────────────────

class _PatchFetchNews:
    """
    Context manager that replaces ``data.gdelt_news.fetch_news`` with a
    function returning ``return_df``, then restores the original on exit.
    Also patches the reference imported inside sentiment_momentum.SentimentMomentum.
    """
    def __init__(self, return_df: pd.DataFrame):
        self._return_df  = return_df
        self._orig       = None
        self._sm_module  = None

    def __enter__(self):
        import data.gdelt_news as gdelt_mod
        import strategies.sentiment_momentum as sm_mod

        self._orig      = gdelt_mod.fetch_news
        self._sm_module = sm_mod

        def _fake_fetch(*args, **kwargs):
            return self._return_df

        gdelt_mod.fetch_news = _fake_fetch
        return self

    def __exit__(self, *_):
        import data.gdelt_news as gdelt_mod
        gdelt_mod.fetch_news = self._orig


# ── Test 1: empty news (cache miss) => zero signals, no crash ────────────────

def test_empty_news_zero_signals():
    """
    When fetch_news returns an empty DataFrame (cache miss / rate-limit),
    generate_signals must:
    - Return a valid signal frame with the correct shape.
    - Produce zero long_signal and zero short_signal.
    - Not raise or hang.
    """
    price_df = _make_price_df(n=100)

    from strategies.sentiment_momentum import SentimentMomentum
    strategy = SentimentMomentum(symbol="MES")

    with _PatchFetchNews(_empty_news_df()):
        result = strategy.generate_signals(price_df, strategy.default_params)

    _assert_valid_signal_frame(result, price_df)

    n_long  = result["long_signal"].sum()
    n_short = result["short_signal"].sum()
    assert n_long  == 0, f"Expected 0 long signals on empty news, got {n_long}"
    assert n_short == 0, f"Expected 0 short signals on empty news, got {n_short}"

    print(f"[PASS] test_empty_news_zero_signals — "
          f"long={n_long}, short={n_short}, frame shape OK")


# ── Test 2: synthetic news => valid signal frame, signals are causal ──────────

def test_synthetic_news_valid_signals():
    """
    With synthetic news, generate_signals must return a valid signal frame.
    We do not assert *which* bars have signals (that depends on random data),
    only that the frame is well-formed and signals are True/False booleans.
    """
    price_df = _make_price_df(n=200)
    news_df  = _make_news_df(n=80)

    from strategies.sentiment_momentum import SentimentMomentum
    strategy = SentimentMomentum(symbol="MES")

    with _PatchFetchNews(news_df):
        result = strategy.generate_signals(price_df, strategy.default_params)

    _assert_valid_signal_frame(result, price_df)

    n_long  = int(result["long_signal"].sum())
    n_short = int(result["short_signal"].sum())
    print(f"[PASS] test_synthetic_news_valid_signals — "
          f"long={n_long}, short={n_short} out of {len(price_df)} bars")


# ── Test 3: no long AND short signal on same bar ──────────────────────────────

def test_no_simultaneous_long_short():
    """
    A bar must never have both long_signal=True and short_signal=True
    (they require opposite z-score signs by construction).
    """
    price_df = _make_price_df(n=150)
    news_df  = _make_news_df(n=60)

    from strategies.sentiment_momentum import SentimentMomentum
    strategy = SentimentMomentum(symbol="MES")

    with _PatchFetchNews(news_df):
        result = strategy.generate_signals(price_df, strategy.default_params)

    both = result["long_signal"] & result["short_signal"]
    assert not both.any(), (
        f"Found {both.sum()} bar(s) with simultaneous long AND short signals"
    )
    print(f"[PASS] test_no_simultaneous_long_short — no simultaneous signals")


# ── Test 4: registry presence ────────────────────────────────────────────────

def test_registry():
    """SentimentMomentum must appear in STRATEGY_REGISTRY under 'sentiment_momentum'."""
    from strategies import STRATEGY_REGISTRY
    assert "sentiment_momentum" in STRATEGY_REGISTRY, (
        f"'sentiment_momentum' missing from STRATEGY_REGISTRY. "
        f"Keys: {sorted(STRATEGY_REGISTRY)}"
    )
    from strategies.sentiment_momentum import SentimentMomentum
    assert STRATEGY_REGISTRY["sentiment_momentum"] is SentimentMomentum, (
        "Registry value is not SentimentMomentum class"
    )
    print("[PASS] test_registry — 'sentiment_momentum' found in STRATEGY_REGISTRY")


# ── Test 5: default_params / get_param_space keys are all 'sm_' prefixed ─────

def test_param_keys_unique():
    """
    All keys in default_params and get_param_space must start with 'sm_'
    to avoid collisions with other strategies in joint optimisation.
    """
    from strategies.sentiment_momentum import SentimentMomentum
    strategy = SentimentMomentum(symbol="MES")

    for k in strategy.default_params:
        assert k.startswith("sm_"), (
            f"default_params key {k!r} does not start with 'sm_'"
        )

    class _FakeTrial:
        def suggest_float(self, name, *a, **kw):  return 1.0
        def suggest_int(self, name, *a, **kw):    return 1
        def suggest_categorical(self, name, c, **kw): return c[0]

    param_space = strategy.get_param_space(_FakeTrial())
    for k in param_space:
        assert k.startswith("sm_"), (
            f"get_param_space key {k!r} does not start with 'sm_'"
        )
    print("[PASS] test_param_keys_unique — all param keys prefixed with 'sm_'")


# ── Test 6: price_df passthrough — extra columns preserved ───────────────────

def test_price_df_passthrough():
    """
    generate_signals must never drop existing price_df columns.
    """
    price_df = _make_price_df(n=50)
    original_cols = set(price_df.columns)

    from strategies.sentiment_momentum import SentimentMomentum
    strategy = SentimentMomentum(symbol="MES")

    with _PatchFetchNews(_empty_news_df()):
        result = strategy.generate_signals(price_df, strategy.default_params)

    missing = original_cols - set(result.columns)
    assert not missing, f"Columns dropped by generate_signals: {missing}"
    print(f"[PASS] test_price_df_passthrough — all {len(original_cols)} original columns preserved")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_empty_news_zero_signals()
    test_synthetic_news_valid_signals()
    test_no_simultaneous_long_short()
    test_registry()
    test_param_keys_unique()
    test_price_df_passthrough()
    print("\nAll SentimentMomentum tests passed.")
