"""
test_engine_costs.py — Phase 1 T2.1-T2.4 + T5.1 regression tests.

Run with:
    python3 tests/test_engine_costs.py

All tests use tiny synthetic OHLCV + signals frames.  No pytest dependency.
"""
from __future__ import annotations

import sys
import os

# Allow running from either the repo root or the tests/ directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import math
import traceback
from typing import List

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(
    opens:  List[float],
    highs:  List[float],
    lows:   List[float],
    closes: List[float],
    n_bars: int | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a UTC DatetimeIndex."""
    n = n_bars or len(opens)
    idx = pd.date_range("2024-01-02", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000},
        index=idx,
    )


def _make_signals(
    n:           int,
    long_bars:   List[int],      # bars where long_signal=True
    entry_price: List[float],
    stop_loss:   List[float],
    take_profit: List[float],
) -> pd.DataFrame:
    """Build a signals DataFrame.  All unlisted bars have no signal."""
    idx = pd.date_range("2024-01-02", periods=n, freq="15min", tz="UTC")
    long_sig  = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)
    ep  = np.full(n, np.nan)
    sl  = np.full(n, np.nan)
    tp  = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=int)

    for bar, e, s, t in zip(long_bars, entry_price, stop_loss, take_profit):
        long_sig[bar] = True
        ep[bar]  = e
        sl[bar]  = s
        tp[bar]  = t
        cnt[bar] = 5   # above MIN_CONFLUENCE_CONFIRMATIONS

    return pd.DataFrame(
        {
            "long_signal":      long_sig,
            "short_signal":     short_sig,
            "entry_price":      ep,
            "stop_loss":        sl,
            "take_profit":      tp,
            "confluence_count": cnt,
        },
        index=idx,
    )


def _engine_zero_cost(symbol: str = "MES", account_size: float = 50_000.0) -> BacktestEngine:
    """Engine with zero slippage and zero commission for clean PnL isolation."""
    return BacktestEngine(
        symbol=symbol,
        account_size=account_size,
        risk_pct=100.0,        # size up to use full equity risk (1 contract minimum)
        commission=0.0,
        slippage_ticks=0,
        news_slippage_ticks=0,
        max_hold_bars=9999,
        max_daily_dd=1.0,      # disable circuit breakers
        max_weekly_dd=1.0,
        max_contracts=500,
        min_confluence=0,
    )


# ── Test registry ─────────────────────────────────────────────────────────────

_PASS: List[str] = []
_FAIL: List[str] = []


def _run(name: str, fn):
    try:
        fn()
        _PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:
        _FAIL.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()


# ── T5.1  Buy-and-hold reproduction ──────────────────────────────────────────

def test_buy_and_hold_reproduction():
    """
    With an always-long signal, zero slippage, and zero commission the engine's
    net PnL must match (close[last] - fill_price) * point_value * contracts.

    To make fill_price == close[0] we set every open == prior close (gap-free
    continuous series).  Then fill_price = open[1] = close[0], so engine PnL
    equals (close[last] - close[0]) * point_value * contracts.
    """
    # 10-bar upward drift; open[i] == close[i-1] (gap-free)
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    opens  = [closes[0]] + closes[:-1]       # open[i] = close[i-1]
    highs  = [c + 0.50 for c in closes]
    lows   = [c - 0.50 for c in closes]
    n = len(closes)

    df      = _make_df(opens, highs, lows, closes)
    # Signal fires at bar 0 so entry is bar 1
    signals = _make_signals(
        n=n,
        long_bars=[0],
        entry_price=[opens[0]],
        stop_loss=[opens[0] - 5.0],   # wide stop — won't be hit
        take_profit=[opens[0] + 500.0],  # unreachable — timeout exit
    )

    engine = _engine_zero_cost()
    # Bypass warmup: the engine slices off WARMUP_BARS from df.
    # We embed warmup_end_idx=0 so no bars are skipped.
    df.attrs["warmup_end_idx"] = 0

    result = engine.run(df, signals)
    assert len(result.trades) == 1, f"Expected 1 trade, got {len(result.trades)}"

    trade = result.trades.iloc[0]
    point_value = engine.point_value  # 5.0 for MES
    contracts   = trade["contracts"]

    # fill_price = open[1] = close[0] (gap-free)
    fill_price = trade["entry_price"]
    expected_bah = (closes[-1] - fill_price) * point_value * contracts
    engine_pnl   = trade["pnl"]

    tol = 1e-4
    assert abs(engine_pnl - expected_bah) < tol, (
        f"Buy-and-hold mismatch: engine={engine_pnl:.6f}, "
        f"expected={expected_bah:.6f}, diff={abs(engine_pnl - expected_bah):.6e}"
    )


# ── T2.2  Cost fixture ────────────────────────────────────────────────────────

def test_cost_fixture():
    """
    One known long trade hitting TP.  Assert:
        net_pnl == gross_pnl - (commission*2*qty) - entry_slip_cost - exit_slip_cost

    Setup (MES: tick_size=0.25, point_value=5.0):
      Bar 0  : signal fires
      Bar 1  : entry open=100.00 + 1 tick slippage (0.25) => fill=100.25
      Bar 2  : high=103.00 > tp=102.00 => TP hit; raw exit=102.00
               exit slippage: -1 tick (0.25) => exit_price=101.75
      SL=99.00 (not hit)

    Gross PnL = (101.75 - 100.25) * 1 * 5 * qty = 7.50 * qty
    Commission = 0.62 * 2 * qty = 1.24 * qty
    Net PnL    = (7.50 - 1.24) * qty = 6.26 * qty
    """
    tick   = 0.25
    tick_v = 1.25
    pv     = 5.0

    entry_open = 100.0
    sl         = 99.0
    tp         = 102.0
    commission = 0.62
    slip_ticks = 1

    # Bar 0: signal; Bar 1: entry (open=100); Bar 2: TP bar (high=103)
    opens  = [99.5,  entry_open, 100.5]
    closes = [99.75, 101.0,     102.5]
    highs  = [100.0, 101.5,     103.0]   # bar 2 high=103 > tp=102 => TP
    lows   = [99.25, 99.75,     100.25]
    n = 3

    df      = _make_df(opens, highs, lows, closes)
    df.attrs["warmup_end_idx"] = 0

    signals = _make_signals(
        n=n,
        long_bars=[0],
        entry_price=[entry_open],
        stop_loss=[sl],
        take_profit=[tp],
    )

    engine = BacktestEngine(
        symbol="MES",
        account_size=50_000.0,
        risk_pct=100.0,
        commission=commission,
        slippage_ticks=slip_ticks,
        news_slippage_ticks=slip_ticks,  # keep uniform for this test
        max_hold_bars=9999,
        max_daily_dd=1.0,
        max_weekly_dd=1.0,
        max_contracts=500,
        min_confluence=0,
    )

    result = engine.run(df, signals)
    assert len(result.trades) == 1, f"Expected 1 trade, got {len(result.trades)}"
    trade = result.trades.iloc[0]

    qty          = trade["contracts"]
    fill_price   = trade["entry_price"]   # 100.00 + 0.25 = 100.25
    exit_price   = trade["exit_price"]    # 102.00 - 0.25 = 101.75

    exp_entry_slip_cost = slip_ticks * tick * qty * pv
    exp_exit_slip_cost  = slip_ticks * tick * qty * pv
    exp_gross    = (tp - exit_price + (tp - fill_price - (tp - 100.25))) * pv * qty
    # Simpler direct calculation:
    exp_gross    = (exit_price - fill_price) * pv * qty
    exp_comm     = commission * 2 * qty
    exp_net      = exp_gross - exp_comm

    actual_net   = trade["pnl"]
    actual_gross = trade["pnl_gross"]
    actual_comm  = trade["commission"]

    tol = 1e-4
    assert abs(actual_gross - exp_gross) < tol, (
        f"Gross PnL mismatch: got {actual_gross:.4f}, expected {exp_gross:.4f}"
    )
    assert abs(actual_comm - exp_comm) < tol, (
        f"Commission mismatch: got {actual_comm:.4f}, expected {exp_comm:.4f}"
    )
    assert abs(actual_net - exp_net) < tol, (
        f"Net PnL mismatch: got {actual_net:.4f}, expected {exp_net:.4f}"
    )
    assert trade["exit_reason"] == "tp", f"Expected TP exit, got {trade['exit_reason']}"

    # Verify fill prices match expectations
    assert abs(fill_price - (entry_open + slip_ticks * tick)) < tol, (
        f"Fill price wrong: {fill_price} != {entry_open + slip_ticks * tick}"
    )
    assert abs(exit_price - (tp - slip_ticks * tick)) < tol, (
        f"Exit price wrong: {exit_price} != {tp - slip_ticks * tick}"
    )


# ── T2.3a  Gap-through stop ───────────────────────────────────────────────────

def test_gap_through_stop():
    """
    A bar that OPENS below the stop for a long trade must fill at the OPEN
    (gap fill), not at the stop price.

    Bar 0: signal fires; Bar 1: entry open=100, fill=100 (zero slippage)
    Bar 2: opens at 97.0 (below SL=98.0) => gap-through, fill at open=97.0
           not at SL=98.0.
    """
    sl = 98.0
    opens  = [99.5,  100.0, 97.0,  96.5]
    closes = [99.75, 99.5,  97.25, 97.0]
    highs  = [100.0, 101.0, 98.5,  97.5]
    lows   = [99.25, 99.0,  96.5,  96.25]
    n = 4

    df      = _make_df(opens, highs, lows, closes)
    df.attrs["warmup_end_idx"] = 0

    signals = _make_signals(
        n=n,
        long_bars=[0],
        entry_price=[100.0],
        stop_loss=[sl],
        take_profit=[110.0],   # unreachable
    )

    engine = _engine_zero_cost()  # zero slippage to isolate gap logic
    result = engine.run(df, signals)

    assert len(result.trades) == 1, f"Expected 1 trade, got {len(result.trades)}"
    trade = result.trades.iloc[0]

    assert trade["exit_reason"] == "sl", f"Expected SL exit, got {trade['exit_reason']}"
    # With zero slippage, the raw exit price (open=97.0) is returned directly
    # because the gap opened below the stop.  Exit price == open[2] == 97.0
    assert abs(trade["exit_price"] - opens[2]) < 1e-4, (
        f"Gap-through: exit_price={trade['exit_price']:.4f}, "
        f"expected open={opens[2]:.4f} (NOT sl={sl:.4f})"
    )
    # Crucially, exit must be WORSE than the stop
    assert trade["exit_price"] < sl, (
        f"Gap-fill must be worse than SL: exit={trade['exit_price']:.4f}, sl={sl:.4f}"
    )


# ── T2.3b  SL-before-TP on same bar ─────────────────────────────────────────

def test_sl_before_tp_same_bar():
    """
    When a single bar's range spans BOTH SL and TP, the engine must record
    an SL exit (pessimistic / worst-case intrabar rule).

    Long trade: fill=100 (zero slippage), SL=98, TP=104.
    Bar 1 (first live bar after entry): low=97 (<= SL) AND high=105 (> TP).
    => SL must be hit first.
    """
    sl = 98.0
    tp = 104.0

    # Bar 0: signal; Bar 1: entry open=100; Bar 2: the both-hit bar
    opens  = [99.5,  100.0, 100.0]
    closes = [99.75, 100.0, 101.0]
    highs  = [100.0, 100.5, 105.0]   # high > TP
    lows   = [99.25, 99.5,  97.0]    # low < SL
    n = 3

    df      = _make_df(opens, highs, lows, closes)
    df.attrs["warmup_end_idx"] = 0

    signals = _make_signals(
        n=n,
        long_bars=[0],
        entry_price=[100.0],
        stop_loss=[sl],
        take_profit=[tp],
    )

    engine = _engine_zero_cost()
    result = engine.run(df, signals)

    assert len(result.trades) == 1, f"Expected 1 trade, got {len(result.trades)}"
    trade = result.trades.iloc[0]

    assert trade["exit_reason"] == "sl", (
        f"Same-bar SL+TP: expected SL exit (pessimistic rule), "
        f"got '{trade['exit_reason']}'"
    )
    # With zero slippage and no gap (open[2]=100 > sl=98), exit price == sl
    assert abs(trade["exit_price"] - sl) < 1e-4, (
        f"Exit price should be at SL={sl}, got {trade['exit_price']:.4f}"
    )


# ── T2.4  Mark-to-market drawdown ─────────────────────────────────────────────

def test_mark_to_market_drawdown():
    """
    A trade that moves sharply against us before recovering to TP must show
    a max_drawdown that reflects the intratrade adverse excursion, not just
    the tiny realized-at-exit loss.

    Long trade (MES, point_value=5): fill=100 (zero slippage/commission)
      Bar 2 (intratrade): close=90 => unrealized loss = (90-100)*5*qty = -50*qty
      Bar 3 (exit):       high=112 > TP=111 => TP hit, raw exit=111, no slip

    The equity curve must dip below account_size during bar 2, so the computed
    max_drawdown must be positive (there was a drawdown during the trade).

    This is the key difference from the old engine, which stamped equity only
    at exit and so showed zero drawdown for this trade.
    """
    account_size = 50_000.0
    tp = 111.0

    # Bar 0: signal; Bar 1: entry open=100; Bar 2: adverse; Bar 3: TP
    opens  = [99.5,  100.0, 95.0,  98.0]
    closes = [99.75, 101.0, 90.0,  112.0]
    highs  = [100.0, 101.5, 96.0,  112.0]   # bar 3 high > tp=111
    lows   = [99.25, 99.5,  88.0,  98.0]
    n = 4

    df      = _make_df(opens, highs, lows, closes)
    df.attrs["warmup_end_idx"] = 0

    signals = _make_signals(
        n=n,
        long_bars=[0],
        entry_price=[100.0],
        stop_loss=[80.0],   # wide stop — won't be hit
        take_profit=[tp],
    )

    engine = _engine_zero_cost(account_size=account_size)
    result = engine.run(df, signals)

    assert len(result.trades) == 1, f"Expected 1 trade, got {len(result.trades)}"
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "tp", f"Expected TP, got {trade['exit_reason']}"

    # The equity curve should dip below account_size at bar 2 (intratrade bar)
    eq = result.equity_curve
    min_eq = eq.min()

    assert min_eq < account_size, (
        f"Mark-to-market: equity curve never dipped below account_size={account_size:.0f}. "
        f"min_equity={min_eq:.2f}. The old engine would not catch intratrade excursion."
    )

    # Compute max_drawdown from the curve directly to confirm it's positive
    rolling_max = eq.cummax()
    dd_series   = (rolling_max - eq) / rolling_max
    max_dd      = dd_series.max()

    assert max_dd > 0.0, (
        f"max_drawdown from equity curve should be > 0 (intratrade adverse move), "
        f"got {max_dd:.6f}"
    )

    # The metrics computed by the engine should also reflect this
    metrics_mdd = result.metrics.get("max_drawdown", 0.0)
    assert metrics_mdd > 0.0, (
        f"metrics['max_drawdown'] should be > 0, got {metrics_mdd:.6f}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== engine cost + fill tests ===\n")

    _run("T5.1  buy-and-hold reproduction",   test_buy_and_hold_reproduction)
    _run("T2.2  cost fixture (gross/comm/net)", test_cost_fixture)
    _run("T2.3a gap-through stop fill",        test_gap_through_stop)
    _run("T2.3b SL-before-TP same bar",        test_sl_before_tp_same_bar)
    _run("T2.4  mark-to-market drawdown",      test_mark_to_market_drawdown)

    print(f"\n{'='*40}")
    print(f"Passed: {len(_PASS)} / {len(_PASS) + len(_FAIL)}")
    if _FAIL:
        print(f"Failed: {_FAIL}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
