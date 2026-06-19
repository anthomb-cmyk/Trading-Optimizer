"""
Vectorized backtesting engine for APEX Optimizer.

Design goals:
  - No Python loops over bars; trade search is vectorized using numpy slicing
  - Single-position mode (one open trade at a time — realistic for MES/MNQ)
  - Realistic fills: enter at next bar's open after signal fires
  - Commission and slippage applied on BOTH entry and exit (round trip)
  - Daily and weekly drawdown circuit breakers
  - Returns typed BacktestResult (trades DataFrame + equity curve + metrics)

Trade lifecycle:
  bar[i]  → signal fires          (strategy detects setup)
  bar[i+1] → entry at open + adverse slippage
  bar[j]  → SL or TP hit; fills are pessimistic (see _find_exit docstring)
           OR max_hold_bars reached → close at close price

Slippage model (T2.1):
  - Normal bars : SLIPPAGE_TICKS_NORMAL ticks each side (entry + exit)
  - High-vol bars: SLIPPAGE_TICKS_NEWS ticks each side (ATR-spike regime)
  High-vol is defined as bar_range > ATR_SPIKE_RATIO * rolling_atr, or an
  explicit "atr_spike" boolean column in the signals frame if present.

Intrabar SL-vs-TP rule (T2.3 / T2.4):
  When a single bar's range spans BOTH the SL and TP simultaneously, the
  STOP is assumed hit first (worst-case / pessimistic assumption). This
  choice is intentionally conservative: in live trading you cannot know
  which level the market visited first within an OHLCV bar, so assuming
  the worst outcome is the correct back-test bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.metrics import compute_all_metrics, compute_score
from config.logger import get_logger
from config.settings import INSTRUMENTS, MIN_CONFLUENCE_CONFIRMATIONS, RISK, WARMUP_BARS
from risk.position_sizer import PositionSizer

try:
    from filters import FilterPipeline as _FilterPipeline
except ImportError:
    _FilterPipeline = None  # type: ignore

log = get_logger(__name__)

# ── Slippage parameters (T2.1) ───────────────────────────────────────────────
# Applied on BOTH entry and exit.  Keep here so callers can override without
# touching settings.py (avoids merge conflicts with the settings module).
SLIPPAGE_TICKS_NORMAL: int = 1    # ticks per side in normal conditions
SLIPPAGE_TICKS_NEWS:   int = 3    # ticks per side on high-volatility / news bars
ATR_SPIKE_RATIO:     float = 2.0  # bar_range > ratio * rolling_atr → high-vol


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    trades:       pd.DataFrame
    equity_curve: pd.Series
    metrics:      Dict[str, float]
    score:        float = 0.0

    def __post_init__(self):
        self.score = self.metrics.get("score", 0.0)

    def is_valid(self, min_trades: int = 20) -> bool:
        return len(self.trades) >= min_trades

    def summary(self) -> str:
        m = self.metrics
        return (
            f"Trades={int(m.get('total_trades', 0)):4d} | "
            f"WR={m.get('win_rate', 0):.1%} | "
            f"PF={m.get('profit_factor', 0):.2f} | "
            f"MDD={m.get('max_drawdown', 0):.1%} | "
            f"Score={m.get('score', 0):.4f}"
        )


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Run a vectorized backtest given a signal DataFrame and market data.

    Accepts signals from either a single strategy or StrategySystem.
    """

    def __init__(
        self,
        symbol:              str   = "MES",
        account_size:        float = RISK["account_size"],
        risk_pct:            float = RISK["max_risk_per_trade_pct"],
        commission:          float = RISK["commission_per_contract"],
        slippage_ticks:      int   = SLIPPAGE_TICKS_NORMAL,
        news_slippage_ticks: int   = SLIPPAGE_TICKS_NEWS,
        max_hold_bars:       int   = 100,
        max_daily_dd:        float = RISK["max_daily_drawdown_pct"] / 100,
        max_weekly_dd:       float = RISK["max_weekly_drawdown_pct"] / 100,
        max_contracts:       int   = 500,
        min_confluence:      int   = MIN_CONFLUENCE_CONFIRMATIONS,
        filters:             Optional[Any] = None,
    ):
        self.symbol              = symbol
        self.instrument          = INSTRUMENTS[symbol]
        self.account_size        = account_size
        self.risk_pct            = risk_pct / 100
        # T2.2: commission_per_contract is applied on EACH side (entry + exit).
        # IB micro (MES/MNQ) round-trip is ~$1.60/contract, so the default 0.62
        # per-side already models that correctly when doubled; if commission is
        # passed as a round-trip figure, halve it before constructing the engine.
        self.commission          = commission
        self.slippage_ticks      = slippage_ticks
        self.news_slippage_ticks = news_slippage_ticks
        self.tick_value          = self.instrument["tick_value"]
        self.tick_size           = self.instrument["tick_size"]
        self.point_value         = self.instrument["point_value"]
        self.max_hold_bars       = max_hold_bars
        self.max_daily_dd        = max_daily_dd
        self.max_weekly_dd       = max_weekly_dd
        self.filters             = filters
        self._sizer = PositionSizer(
            point_value  = self.point_value,
            tick_size    = self.tick_size,
            risk_pct     = self.risk_pct,
            max_contracts = max_contracts,
        )
        self._min_confluence = min_confluence

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        df:      pd.DataFrame,
        signals: pd.DataFrame,
        params:  Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """
        Execute the backtest.

        df      : market data with OHLCV columns (DatetimeIndex UTC)
        signals : output of strategy.generate_signals() or system.generate_signals()
        params  : unused here but passed through for reference
        """
        # Skip warmup bars
        warmup = df.attrs.get("warmup_end_idx", WARMUP_BARS)
        df      = df.iloc[warmup:].copy()
        signals = signals.iloc[warmup:].copy()

        if len(df) < 2:
            return self._empty_result()

        # Align signals to df
        signals = signals.reindex(df.index)

        # Apply Phase-4 filters (news blackout + volatility circuit breaker)
        if self.filters is not None:
            signals = self.filters.apply(df, signals)

        trades_list, equity_curve = self._simulate(df, signals)

        if not trades_list:
            return self._empty_result()

        trades = pd.DataFrame(trades_list)
        eq_s   = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
        metrics = compute_all_metrics(trades, eq_s)
        return BacktestResult(trades=trades, equity_curve=eq_s, metrics=metrics)

    # ── Core simulation ───────────────────────────────────────────────────────

    def _simulate(
        self,
        df:      pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Tuple[list, list]:
        """
        Iterate over entry signals; for each trade find exit using numpy.
        Returns (trades_list, equity_curve_list).

        Slippage (T2.1): applied adversely on BOTH entry and exit.
        Commission (T2.2): applied per-side (entry + exit = round trip × qty).
        Fills (T2.3): pessimistic — see _find_exit docstring.
        Equity (T2.4): mark-to-market every bar inside an open trade using
            unrealized PnL = (close - fill_price) * direction * point_value * qty
        """
        opens   = df["open"].values
        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values
        dates   = df.index

        long_sig  = signals["long_signal"].fillna(False).values.astype(bool)
        short_sig = signals["short_signal"].fillna(False).values.astype(bool)
        entry_px  = signals["entry_price"].values
        sl_px     = signals["stop_loss"].values
        tp_px     = signals["take_profit"].values
        conf_cnt  = signals.get("confluence_count", pd.Series(0, index=df.index)).values

        # T2.1: detect high-volatility bars.  Prefer an explicit "atr_spike"
        # boolean column when the strategy provides it; otherwise compute
        # bar_range / rolling_atr and flag bars where ratio > ATR_SPIKE_RATIO.
        if "atr_spike" in signals.columns:
            high_vol = signals["atr_spike"].fillna(False).values.astype(bool)
        elif "atr" in df.columns:
            bar_range = highs - lows
            atr_vals  = df["atr"].values
            # avoid division by zero on zero-ATR bars
            safe_atr  = np.where(atr_vals > 0, atr_vals, np.nan)
            high_vol  = (bar_range / safe_atr) > ATR_SPIKE_RATIO
            high_vol  = np.where(np.isnan(high_vol), False, high_vol).astype(bool)
        else:
            # No ATR available: compute a simple 14-bar rolling range proxy
            bar_range  = highs - lows
            roll_range = pd.Series(bar_range).rolling(14, min_periods=1).mean().values
            safe_roll  = np.where(roll_range > 0, roll_range, np.nan)
            high_vol   = (bar_range / safe_roll) > ATR_SPIKE_RATIO
            high_vol   = np.where(np.isnan(high_vol), False, high_vol).astype(bool)

        n             = len(df)
        equity        = self.account_size
        equity_curve  = [equity] * n
        trades        = []

        # Daily / weekly drawdown tracking
        daily_start_equity  = equity
        weekly_start_equity = equity
        halted_until_bar    = 0   # index after which trading resumes

        i = 0
        while i < n - 1:
            # ── Drawdown circuit breaker ──────────────────────────────────────
            if i >= halted_until_bar:
                # Daily reset
                if i > 0 and dates[i].date() != dates[i - 1].date():
                    daily_start_equity = equity
                # Weekly reset (Monday)
                if i > 0 and dates[i].weekday() == 0 and dates[i - 1].weekday() != 0:
                    weekly_start_equity = equity

                daily_dd  = (daily_start_equity  - equity) / daily_start_equity
                weekly_dd = (weekly_start_equity - equity) / weekly_start_equity

                if daily_dd >= self.max_daily_dd:
                    halted_until_bar = self._next_day_bar(dates, i)
                    i += 1
                    continue
                if weekly_dd >= self.max_weekly_dd:
                    halted_until_bar = self._next_week_bar(dates, i)
                    i += 1
                    continue
            else:
                i += 1
                continue

            # ── Entry check ───────────────────────────────────────────────────
            is_long  = long_sig[i]  and not np.isnan(entry_px[i])
            is_short = short_sig[i] and not np.isnan(entry_px[i])

            if not (is_long or is_short):
                i += 1
                continue

            direction   = 1 if is_long else -1
            sl          = sl_px[i]
            tp          = tp_px[i]
            signal_conf = int(conf_cnt[i])

            # T2.1: choose slippage size based on volatility regime at entry bar.
            entry_bar  = i + 1
            entry_hvol = high_vol[i]  # regime flag at signal bar
            entry_slip_ticks = (
                self.news_slippage_ticks if entry_hvol else self.slippage_ticks
            )
            # Adverse entry slippage: longs pay more, shorts receive less
            entry_slippage = entry_slip_ticks * self.tick_size * direction
            fill_price     = opens[entry_bar] + entry_slippage

            # Validate SL/TP are sensible after slippage
            if direction == 1  and (sl >= fill_price or tp <= fill_price):
                i += 1
                continue
            if direction == -1 and (sl <= fill_price or tp >= fill_price):
                i += 1
                continue

            # ── Position sizing ───────────────────────────────────────────────
            sl_dist_pts = abs(fill_price - sl)
            if sl_dist_pts < self.tick_size:
                i += 1
                continue
            contracts = self._sizer.get_contracts(
                equity           = equity,
                sl_distance_pts  = sl_dist_pts,
                confluence_count = signal_conf,
                min_confluence   = self._min_confluence,
            )

            # ── Find exit (T2.3 pessimistic fills) ───────────────────────────
            max_exit = min(entry_bar + self.max_hold_bars, n - 1)
            exit_bar, exit_price_raw, exit_reason = self._find_exit(
                highs, lows, closes, opens,
                entry_bar, max_exit,
                fill_price, sl, tp, direction,
            )

            # T2.1: adverse exit slippage (regime at exit bar)
            exit_hvol = high_vol[exit_bar]
            exit_slip_ticks = (
                self.news_slippage_ticks if exit_hvol else self.slippage_ticks
            )
            if exit_reason == "timeout":
                # Timeout: close at close price; slippage still applies
                exit_slippage = exit_slip_ticks * self.tick_size * (-direction)
                exit_price    = exit_price_raw + exit_slippage
            elif exit_reason == "sl":
                # SL exits already incorporate gap-fill at open (see _find_exit);
                # additionally apply exit slippage in the adverse direction
                exit_slippage = exit_slip_ticks * self.tick_size * (-direction)
                exit_price    = exit_price_raw + exit_slippage
            else:
                # TP (limit order): slippage is negligible on a limit, but we
                # still apply it for consistency (conservative).
                exit_slippage = exit_slip_ticks * self.tick_size * (-direction)
                exit_price    = exit_price_raw + exit_slippage

            # ── PnL calculation (T2.2: commission per side × qty) ─────────────
            # commission_per_contract is charged per side (entry + exit).
            # IB micro round-trip is ~$1.60/contract; default 0.62/side = $1.24
            # round-trip — may be slightly low but reflects NinjaTrader defaults.
            gross_pnl  = (exit_price - fill_price) * direction * self.point_value * contracts
            comm_entry = self.commission * contracts   # entry side
            comm_exit  = self.commission * contracts   # exit side
            total_comm = comm_entry + comm_exit
            net_pnl    = gross_pnl - total_comm
            equity    += net_pnl

            # T2.4: mark-to-market equity inside the trade.
            # Bars during the trade carry unrealized PnL = (close - fill_price)
            # * direction * point_value * contracts.  The last bar (exit_bar)
            # carries the realized equity already computed above.
            for b in range(entry_bar, exit_bar):
                unrealized = (
                    (closes[b] - fill_price) * direction
                    * self.point_value * contracts
                    - total_comm   # deduct full commission at start (conservative)
                )
                equity_curve[b] = self.account_size + sum(
                    t["pnl"] for t in trades
                ) + unrealized
            # Exit bar: realized equity
            equity_curve[exit_bar] = equity

            sl_dist_realized = abs(exit_price - fill_price)
            rr_realized      = (sl_dist_realized / sl_dist_pts) * (1 if exit_reason == "tp" else -1)

            trades.append({
                "entry_bar":      entry_bar,
                "exit_bar":       exit_bar,
                "entry_date":     dates[entry_bar],
                "exit_date":      dates[exit_bar],
                "direction":      direction,
                "entry_price":    round(fill_price, 4),
                "exit_price":     round(exit_price, 4),
                "stop_loss":      round(sl, 4),
                "take_profit":    round(tp, 4),
                "contracts":      contracts,
                "pnl":            round(net_pnl, 2),
                "pnl_gross":      round(gross_pnl, 2),
                "commission":     round(total_comm, 2),
                "exit_reason":    exit_reason,
                "rr_realized":    round(rr_realized, 3),
                "duration_bars":  exit_bar - entry_bar,
                "equity_after":   round(equity, 2),
                "confluence_count": signal_conf,
            })

            i = exit_bar + 1   # advance past this trade — one position at a time

        return trades, equity_curve

    # ── Exit finder (vectorized per trade) ───────────────────────────────────

    @staticmethod
    def _find_exit(
        highs:      np.ndarray,
        lows:       np.ndarray,
        closes:     np.ndarray,
        opens:      np.ndarray,
        entry_bar:  int,
        max_bar:    int,
        fill_price: float,
        sl:         float,
        tp:         float,
        direction:  int,
    ) -> Tuple[int, float, str]:
        """
        Vectorized exit with pessimistic fill logic (T2.3).

        Rules applied in order:

        1. Gap-through stop: if the bar OPENS beyond the SL level (i.e. the
           market gapped past the stop), fill at the OPEN price (worse than SL).

        2. SL triggered: bar low <= SL (long) or bar high >= SL (short).
           Fill at SL (the gap case above supersedes this when open is worse).

        3. TP triggered: bar must TRADE THROUGH the TP by at least 1 tick,
           not merely touch it.  Strict inequality (> for longs, < for shorts).

        4. Same-bar SL + TP: when a single bar's range spans BOTH levels, the
           STOP is assumed hit first (pessimistic / worst-case intrabar rule).
           The engine cannot know intrabar order from OHLCV data; choosing the
           stop is deliberately conservative.

        5. Timeout: close at the last bar's close price.

        Note: exit slippage is NOT applied here; the caller (_simulate) adds
        adverse exit slippage on top of the raw price returned here.
        """
        fut_h = highs[entry_bar : max_bar + 1]
        fut_l = lows[entry_bar  : max_bar + 1]
        fut_o = opens[entry_bar : max_bar + 1]

        if direction == 1:  # long
            # Gap-through: open below SL (open < SL)
            gap_sl   = fut_o < sl
            # Normal SL touch (low <= SL)
            sl_touch = fut_l <= sl
            sl_hits  = gap_sl | sl_touch
            # TP: bar must trade STRICTLY THROUGH tp (high > tp by >= 1 tick)
            # Using strict > so a bar that merely touches the level does not fill
            tp_hits  = fut_h > tp
        else:  # short
            # Gap-through: open above SL (open > SL)
            gap_sl   = fut_o > sl
            # Normal SL touch (high >= SL)
            sl_touch = fut_h >= sl
            sl_hits  = gap_sl | sl_touch
            # TP: bar must trade STRICTLY THROUGH tp (low < tp)
            tp_hits  = fut_l < tp

        first_sl = int(np.argmax(sl_hits)) if sl_hits.any() else len(fut_h)
        first_tp = int(np.argmax(tp_hits)) if tp_hits.any() else len(fut_h)

        # Intrabar rule: if SL and TP are on the same bar, SL is hit first
        if first_sl <= first_tp and sl_hits.any():
            abs_bar = entry_bar + first_sl
            # Gap-through fill: if the bar opened past the stop, fill at open
            if direction == 1 and opens[abs_bar] < sl:
                return abs_bar, opens[abs_bar], "sl"
            if direction == -1 and opens[abs_bar] > sl:
                return abs_bar, opens[abs_bar], "sl"
            return abs_bar, sl, "sl"

        if tp_hits.any():
            return entry_bar + first_tp, tp, "tp"

        # Timeout — close at last bar's close
        timeout_bar = min(entry_bar + len(fut_h) - 1, max_bar)
        return timeout_bar, closes[timeout_bar], "timeout"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _next_day_bar(dates: pd.DatetimeIndex, current: int) -> int:
        current_date = dates[current].date()
        for j in range(current + 1, len(dates)):
            if dates[j].date() > current_date:
                return j
        return len(dates)

    @staticmethod
    def _next_week_bar(dates: pd.DatetimeIndex, current: int) -> int:
        current_week = dates[current].isocalendar().week
        for j in range(current + 1, len(dates)):
            if dates[j].isocalendar().week != current_week:
                return j
        return len(dates)

    def _empty_result(self) -> BacktestResult:
        trades = pd.DataFrame(columns=[
            "entry_bar", "exit_bar", "entry_date", "exit_date",
            "direction", "entry_price", "exit_price", "stop_loss",
            "take_profit", "contracts", "pnl", "pnl_gross", "commission",
            "exit_reason", "rr_realized", "duration_bars",
            "equity_after", "confluence_count",
        ])
        eq = pd.Series([self.account_size], dtype=float)
        metrics = compute_all_metrics(trades, eq)
        return BacktestResult(trades=trades, equity_curve=eq, metrics=metrics)
