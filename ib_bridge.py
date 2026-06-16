"""
IB Live Bridge — multi-symbol, independent per-symbol engines.

Each symbol runs its own StrategySystem, position tracker, and order manager.
All symbols share a single IB connection but otherwise operate independently.

Usage (via main.py):
    python main.py live --symbols MES,MNQ,MGC --paper

Position management
-------------------
Entry  : limit order at strategy signal price (full quantity)
SL     : stop order at strategy stop-loss price (full quantity, parentId-linked)
TP1    : limit order at 1:1 R:R, 30 % of position (parentId-linked)
         → on fill: SL immediately cancelled and replaced at breakeven (entry price)
TP2    : limit order at 2:1 R:R, 40 % of position (parentId-linked)
         → on fill: SL cancelled, $1 trailing stop placed for remaining 30 %
TP3    : limit order at 3:1 R:R, remaining 30 % (parentId-linked)

All TP/SL child orders share the entry's parentId so they only activate after
the entry fills.  The TP1 and TP2 fill-event callbacks fire immediately inside
ib_insync's event loop — no polling delay.

Daily rules (US/Eastern)
------------------------
  Mon–Thu : no new entries after 15:30; all positions closed at 15:45
  Friday  : no new entries after 15:30; all positions closed at 15:30
"""
import asyncio
import json
import sys
from datetime import datetime, time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from config.logger import get_logger
from config.settings import FUTURES_RISK_PER_TRADE, IB as IB_CFG, INSTRUMENTS, RISK
from trade_logger import _logger as _trade_log

log = get_logger("ib_bridge")

_EST = pytz.timezone("America/New_York")

# ── Futures contract specifications ───────────────────────────────────────────
_CONTRACT_SPECS = {
    "MES": {"exchange": "CME",   "point_value": 5.0},
    "MNQ": {"exchange": "CME",   "point_value": 2.0},
    "MGC": {"exchange": "COMEX", "point_value": 10.0},
}
_ROLL_DAYS_BEFORE_EXPIRY = 5   # switch to next contract this many days before expiry

# ── Time-of-day trading rules ─────────────────────────────────────────────────
_ENTRY_CUTOFF_ALL = _time(15, 30)   # no new entries at or after this time (all days)
_EOD_CLOSE_NORMAL = _time(15, 45)   # flatten all positions, Mon–Thu
_EOD_CLOSE_FRIDAY = _time(15, 30)   # flatten all positions on Friday


def _entries_blocked(now: datetime) -> Tuple[bool, str]:
    """Return (blocked, reason) — True when new entries must not be opened."""
    if now.time() >= _ENTRY_CUTOFF_ALL:
        return True, f"entry cutoff {_ENTRY_CUTOFF_ALL.strftime('%H:%M')} EST reached"
    return False, ""


def _eod_close_due(now: datetime) -> Tuple[bool, str]:
    """Return (close_due, reason) — True when all open positions must be closed."""
    t         = now.time()
    is_friday = now.weekday() == 4      # Monday=0 … Friday=4
    if is_friday and t >= _EOD_CLOSE_FRIDAY:
        return True, f"Friday {_EOD_CLOSE_FRIDAY.strftime('%H:%M')} EST — closing before weekend"
    if not is_friday and t >= _EOD_CLOSE_NORMAL:
        return True, f"EOD {_EOD_CLOSE_NORMAL.strftime('%H:%M')} EST — closing before market close"
    return False, ""


# ── Per-symbol engine ─────────────────────────────────────────────────────────

class SymbolEngine:
    """
    Per-symbol signal generator, position tracker, and order manager.

    Partial-TP split: 30 % @ 1:1, 40 % @ 2:1, 30 % @ 3:1.
    When full_qty < 3 (can't split meaningfully) the position is treated as a
    single unit and exits fully at 2:1 R:R.
    """

    TP1_PCT = 0.30
    TP2_PCT = 0.40
    # TP3 gets the remainder (≈ 30 %)

    def __init__(self, symbol: str, params: dict, paper: bool = True):
        from strategies.system import StrategySystem

        self.symbol     = symbol
        self.params     = params
        self.paper      = paper
        self.system     = StrategySystem(symbol=symbol)
        self.instrument = INSTRUMENTS[symbol]

        self._reset_position()
        self.bars_buffer: Optional[pd.DataFrame] = None

    # ── Position state ────────────────────────────────────────────────────────

    def _reset_position(self) -> None:
        """Clear all position and order-tracking state."""
        self.position_side: int             = 0   # 0=flat, 1=long, -1=short
        self.entry_price:   Optional[float] = None
        self.initial_sl:    Optional[float] = None
        self.full_qty:      int             = 0

        # Leg quantities (0 means that leg is skipped)
        self.tp1_qty: int = 0
        self.tp2_qty: int = 0
        self.tp3_qty: int = 0

        # Phase flags (set by fill-event callbacks)
        self.tp1_hit: bool = False
        self.tp2_hit: bool = False

        # Live ib_insync Trade objects
        self.sl_trade  = None
        self.tp1_trade = None
        self.tp2_trade = None
        self.tp3_trade = None

        # Entry metadata for trade logging (populated when bracket is placed)
        self.entry_time:       Optional[datetime] = None
        self.strategy_name:    str                = ""
        self.confluence_score: int                = 0
        self.tp1_price:        Optional[float]    = None
        self.tp2_price:        Optional[float]    = None
        self.tp3_price:        Optional[float]    = None

        # Set True by TP3/SL fill callbacks so poll() can distinguish a callback-logged
        # exit from a manual TWS close (where no callback fires).
        self._final_exit_logged: bool = False

    # ── Contract factory ──────────────────────────────────────────────────────

    def make_contract(self):
        from ib_insync import Future, Stock
        if self.symbol in _CONTRACT_SPECS:
            spec = _CONTRACT_SPECS[self.symbol]
            return Future(self.symbol, exchange=spec["exchange"], currency="USD")
        # ETFs (paper mode only)
        return Stock(self.symbol, "SMART", "USD")

    # ── Data management ───────────────────────────────────────────────────────

    def load_warmup(self) -> None:
        """Prime the indicator buffer with recent Yahoo bars."""
        from data.loader import load_bars
        log.info("[%s] Loading warmup bars…", self.symbol)
        try:
            df = load_bars(self.symbol, "15m", use_cache=False)
            self.bars_buffer = df
            log.info("[%s] Warmup complete: %d bars.", self.symbol, len(df))
        except Exception as exc:
            log.error("[%s] Warmup failed: %s", self.symbol, exc)

    def _ib_bars_to_raw(self, ib_bars) -> pd.DataFrame:
        rows, timestamps = [], []
        for b in ib_bars:
            ts = pd.Timestamp(b.date)
            ts = _EST.localize(ts).astimezone(pytz.utc) if ts.tzinfo is None \
                 else ts.tz_convert("UTC")
            timestamps.append(ts)
            rows.append({"open": b.open, "high": b.high, "low": b.low,
                         "close": b.close, "volume": b.volume})
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps, name="datetime"))

    def _update_buffer(self, ib_bars) -> Optional[pd.DataFrame]:
        from data.loader import _add_indicators
        new_raw = self._ib_bars_to_raw(ib_bars)
        if new_raw.empty:
            return self.bars_buffer
        raw_cols = ["open", "high", "low", "close", "volume"]
        base = self.bars_buffer[raw_cols] if (
            self.bars_buffer is not None and not self.bars_buffer.empty
        ) else pd.DataFrame(columns=raw_cols)
        combined = (
            pd.concat([base, new_raw[raw_cols]])
            .pipe(lambda df: df[~df.index.duplicated(keep="last")])
            .sort_index()
            .tail(500)
        )
        self.bars_buffer = _add_indicators(combined)
        return self.bars_buffer

    # ── Risk / sizing ─────────────────────────────────────────────────────────

    def _calc_qty(self, entry: float, stop: float) -> int:
        """qty = FUTURES_RISK_PER_TRADE / (SL_points × point_value)"""
        risk_per_contract = abs(entry - stop) * self.instrument["point_value"]
        if risk_per_contract <= 0:
            return 0
        return max(int(FUTURES_RISK_PER_TRADE / risk_per_contract), 1)

    def _split_qty(self, full_qty: int) -> Tuple[int, int, int]:
        """
        Split full_qty 30 / 40 / 30 across three TP legs.
        Returns (tp1_qty, tp2_qty, tp3_qty).
        If full_qty < 3 returns (0, 0, full_qty) — single exit at TP2 level.
        """
        if full_qty < 3:
            return 0, 0, full_qty
        tp1 = max(1, round(full_qty * self.TP1_PCT))
        tp2 = max(1, round(full_qty * self.TP2_PCT))
        tp3 = max(1, full_qty - tp1 - tp2)
        return tp1, tp2, tp3

    # ── EOD / weekend forced close ────────────────────────────────────────────

    def close_position(self, ib, contract, reason: str = "") -> None:
        """Cancel all open orders for this symbol and flatten with a market order."""
        from ib_insync import MarketOrder

        if self.position_side == 0:
            return

        for trade in ib.openTrades():
            if trade.contract.symbol == self.symbol:
                try:
                    ib.cancelOrder(trade.order)
                except Exception as exc:
                    log.warning("[%s] Cancel order %s failed: %s",
                                self.symbol, trade.order.orderId, exc)

        # Use actual IB position size (may be reduced by partial TPs already hit)
        remaining_qty = next(
            (abs(int(p.position)) for p in ib.positions()
             if p.contract.symbol == self.symbol),
            self.full_qty,
        )

        if remaining_qty > 0:
            close_action = "SELL" if self.position_side == 1 else "BUY"

            # Capture state now — _reset_position() runs right after placeOrder
            _sym       = self.symbol
            _dir       = "LONG" if self.position_side == 1 else "SHORT"
            _etime     = self.entry_time or datetime.now(_EST)
            _eprice    = self.entry_price or 0.0
            _sl        = self.initial_sl or 0.0
            _strat     = self.strategy_name
            _conf      = self.confluence_score
            _tp1       = self.tp1_price or 0.0
            _exit_rsn  = "EOD" if not reason else ("EOD" if "EOD" in reason.upper() else "MANUAL")

            try:
                mkt_trade = ib.placeOrder(contract, MarketOrder(close_action, remaining_qty))
                log.info("[%s] Closing %d @ market%s",
                         self.symbol, remaining_qty,
                         f" — {reason}" if reason else "")

                def _on_eod_fill(trade, fill):
                    fill_px  = fill.execution.price
                    fill_qty = int(fill.execution.shares)
                    log.info("[%s] %s close filled at %.4f — %d shares",
                             _sym, _exit_rsn, fill_px, fill_qty)
                    _trade_log.log_trade(
                        symbol=_sym,
                        direction=_dir,
                        entry_time=_etime,
                        entry_price=_eprice,
                        exit_time=datetime.now(_EST),
                        exit_price=fill_px,
                        quantity=fill_qty,
                        strategy_name=_strat,
                        confluence_score=_conf,
                        stop_loss=_sl,
                        take_profit=_tp1,
                        exit_reason=_exit_rsn,
                    )

                mkt_trade.fillEvent += _on_eod_fill
            except Exception as exc:
                log.error("[%s] EOD close order failed: %s", self.symbol, exc)

        self._reset_position()

    def _force_close_ib(self, ib, contract, reason: str = "") -> None:
        """Force-close any open IB position regardless of engine state.

        Used when close_due is True but position_side == 0 (engine/IB desync).
        """
        from ib_insync import MarketOrder

        ib_pos = next(
            (p for p in ib.positions() if p.contract.symbol == self.symbol),
            None,
        )
        if ib_pos is None or ib_pos.position == 0:
            log.info("[%s] _force_close_ib: no IB position found — already flat.", self.symbol)
            return

        qty          = abs(int(ib_pos.position))
        side         = 1 if ib_pos.position > 0 else -1
        close_action = "SELL" if side == 1 else "BUY"

        for trade in ib.openTrades():
            if trade.contract.symbol == self.symbol:
                try:
                    ib.cancelOrder(trade.order)
                except Exception as exc:
                    log.warning("[%s] Cancel order %s failed: %s",
                                self.symbol, trade.order.orderId, exc)

        try:
            ib.placeOrder(contract, MarketOrder(close_action, qty))
            log.info("[%s] Force-close: %s %d @ market%s",
                     self.symbol, close_action, qty,
                     f" — {reason}" if reason else "")
        except Exception as exc:
            log.error("[%s] Force-close order failed: %s", self.symbol, exc)

        self._reset_position()

    # ── Trade logging helpers ─────────────────────────────────────────────────

    def _log_exit(
        self,
        exit_price:  float,
        quantity:    int,
        exit_reason: str,
        take_profit: float,
    ) -> None:
        """Log a single closed leg to the trade CSV using current position state."""
        _trade_log.log_trade(
            symbol=self.symbol,
            direction="LONG" if self.position_side == 1 else "SHORT",
            entry_time=self.entry_time or datetime.now(_EST),
            entry_price=self.entry_price or 0.0,
            exit_time=datetime.now(_EST),
            exit_price=exit_price,
            quantity=quantity,
            strategy_name=self.strategy_name,
            confluence_score=self.confluence_score,
            stop_loss=self.initial_sl or 0.0,
            take_profit=take_profit,
            exit_reason=exit_reason,
        )

    def _wire_sl_fill_log(self, sl_trade_obj, intended_tp: float) -> None:
        """Attach a fill-event logger to any SL / trail order, capturing state now."""
        _sym         = self.symbol
        _dir         = "LONG" if self.position_side == 1 else "SHORT"
        _etime       = self.entry_time or datetime.now(_EST)
        _eprice      = self.entry_price or 0.0
        _sl          = self.initial_sl or 0.0
        _strat       = self.strategy_name
        _conf        = self.confluence_score
        _tp          = intended_tp
        _self_engine = self  # captured for closure to mark exit logged

        def _on_fill(trade, fill):
            fill_px  = fill.execution.price
            fill_qty = int(fill.execution.shares)
            log.info("[%s] SL/Trail filled at %.4f — %d shares", _sym, fill_px, fill_qty)
            _trade_log.log_trade(
                symbol=_sym,
                direction=_dir,
                entry_time=_etime,
                entry_price=_eprice,
                exit_time=datetime.now(_EST),
                exit_price=fill_px,
                quantity=fill_qty,
                strategy_name=_strat,
                confluence_score=_conf,
                stop_loss=_sl,
                take_profit=_tp,
                exit_reason="SL",
            )
            _self_engine._final_exit_logged = True

        sl_trade_obj.fillEvent += _on_fill

    # ── Order placement — partial TPs with parentId bracket ──────────────────

    async def _enter_and_place_tps(
        self, ib, contract,
        action: str, full_qty: int,
        entry: float, initial_sl: float,
        strategy_name: str = "",
        confluence_score: int = 0,
    ) -> None:
        """
        Place a parentId-linked bracket:
          parent  : limit entry (full_qty)
          children: SL stop, TP1 limit, TP2 limit, TP3 limit

        Children only activate after the parent (entry) fills.
        Fill-event callbacks on TP1 and TP2 handle SL adjustments immediately.
        """
        from ib_insync import LimitOrder, StopOrder, Order

        direction    = 1 if action == "BUY" else -1
        close_action = "SELL" if direction == 1 else "BUY"
        risk         = abs(entry - initial_sl)

        # ── TP price levels ───────────────────────────────────────────────────
        tp1_price = round(entry + direction * 1.0 * risk, 2)
        tp2_price = round(entry + direction * 2.0 * risk, 2)
        tp3_price = round(entry + direction * 3.0 * risk, 2)

        tp1_qty, tp2_qty, tp3_qty = self._split_qty(full_qty)

        # When full_qty < 3: skip partials, exit full at TP2 level
        if tp1_qty == 0:
            tp3_price = tp2_price

        log.info(
            "[%s] %s %d @ %.2f  SL=%.2f  "
            "TP1=%.2f×%d  TP2=%.2f×%d  TP3=%.2f×%d",
            self.symbol, action, full_qty, entry, initial_sl,
            tp1_price, tp1_qty, tp2_price, tp2_qty, tp3_price, tp3_qty,
        )

        # ── Parent: limit entry ───────────────────────────────────────────────
        entry_order          = LimitOrder(action, full_qty, round(entry, 2))
        entry_order.transmit = False
        entry_trade          = ib.placeOrder(contract, entry_order)
        parent_id            = entry_trade.order.orderId
        log.info("[%s] Entry order placed: orderId=%d  action=%s  qty=%d  limitPx=%.2f",
                 self.symbol, parent_id, action, full_qty, entry)

        # ── SL child (full qty, only live after entry fills) ──────────────────
        sl_order          = StopOrder(close_action, full_qty, round(initial_sl, 2))
        sl_order.parentId = parent_id
        sl_order.transmit = False
        sl_trade          = ib.placeOrder(contract, sl_order)
        log.info("[%s] SL order placed:    orderId=%d  parentId=%d  qty=%d  stopPx=%.2f",
                 self.symbol, sl_trade.order.orderId, parent_id, full_qty, initial_sl)

        # ── TP children ───────────────────────────────────────────────────────
        tp1_trade = tp2_trade = None

        if tp1_qty > 0:
            tp1_order          = LimitOrder(close_action, tp1_qty, tp1_price)
            tp1_order.parentId = parent_id
            tp1_order.transmit = False
            tp1_trade          = ib.placeOrder(contract, tp1_order)
            log.info("[%s] TP1 order placed:   orderId=%d  parentId=%d  qty=%d  limitPx=%.2f",
                     self.symbol, tp1_trade.order.orderId, parent_id, tp1_qty, tp1_price)

        if tp2_qty > 0:
            tp2_order          = LimitOrder(close_action, tp2_qty, tp2_price)
            tp2_order.parentId = parent_id
            tp2_order.transmit = False
            tp2_trade          = ib.placeOrder(contract, tp2_order)
            log.info("[%s] TP2 order placed:   orderId=%d  parentId=%d  qty=%d  limitPx=%.2f",
                     self.symbol, tp2_trade.order.orderId, parent_id, tp2_qty, tp2_price)

        # TP3 — last child, transmit=True activates the entire bracket
        tp3_order          = LimitOrder(close_action, tp3_qty, tp3_price)
        tp3_order.parentId = parent_id
        tp3_order.transmit = True
        tp3_trade          = ib.placeOrder(contract, tp3_order)
        log.info("[%s] TP3 order placed:   orderId=%d  parentId=%d  qty=%d  limitPx=%.2f  [bracket transmitted]",
                 self.symbol, tp3_trade.order.orderId, parent_id, tp3_qty, tp3_price)

        log.info(
            "[%s] ALL bracket order IDs — entry=%d  sl=%d  tp1=%s  tp2=%s  tp3=%d",
            self.symbol,
            parent_id,
            sl_trade.order.orderId,
            str(tp1_trade.order.orderId) if tp1_trade else "SKIPPED(qty<3)",
            str(tp2_trade.order.orderId) if tp2_trade else "SKIPPED(qty<3)",
            tp3_trade.order.orderId,
        )

        # ── Persist state ─────────────────────────────────────────────────────
        self.position_side    = direction
        self.entry_price      = entry
        self.initial_sl       = initial_sl
        self.full_qty         = full_qty
        self.tp1_qty          = tp1_qty
        self.tp2_qty          = tp2_qty
        self.tp3_qty          = tp3_qty
        self.sl_trade         = sl_trade
        self.tp1_trade        = tp1_trade
        self.tp2_trade        = tp2_trade
        self.tp3_trade        = tp3_trade
        self.entry_time       = datetime.now(_EST)
        self.strategy_name    = strategy_name
        self.confluence_score = confluence_score
        self.tp1_price        = tp1_price
        self.tp2_price        = tp2_price
        self.tp3_price        = tp3_price

        # ── TP1 fill callback: IMMEDIATE breakeven SL move ────────────────────
        if tp1_trade is not None:
            def _on_tp1_fill(trade, fill):
                if self.tp1_hit:
                    return
                self.tp1_hit = True
                fill_px = fill.execution.price
                be_qty  = tp2_qty + tp3_qty
                log.info(
                    "[%s] TP1 filled at %.2f — closing %d shares (30%%), moving SL to breakeven %.2f",
                    self.symbol, fill_px, tp1_qty, self.entry_price,
                )
                self._log_exit(fill_px, tp1_qty, "TP1", tp1_price)
                try:
                    ib.cancelOrder(self.sl_trade.order)
                except Exception as exc:
                    log.warning("[%s] Cancel old SL failed: %s", self.symbol, exc)
                be_sl          = StopOrder(close_action, be_qty,
                                           round(self.entry_price, 2))
                self.sl_trade  = ib.placeOrder(contract, be_sl)
                self._wire_sl_fill_log(self.sl_trade, tp2_price)
                log.info("[%s] Breakeven SL live @ %.2f  qty=%d",
                         self.symbol, self.entry_price, be_qty)

            log.info("[%s] TP1 fillEvent wired for orderId %d",
                     self.symbol, tp1_trade.order.orderId)
            tp1_trade.fillEvent += _on_tp1_fill

        # ── TP2 fill callback: activate $1 trailing stop ──────────────────────
        if tp2_trade is not None:
            def _on_tp2_fill(trade, fill):
                if self.tp2_hit:
                    return
                self.tp2_hit = True
                fill_px = fill.execution.price
                log.info(
                    "[%s] TP2 filled @ %.2f — activating $1 trailing stop for %d shares",
                    self.symbol, fill_px, tp3_qty,
                )
                self._log_exit(fill_px, tp2_qty, "TP2", tp2_price)
                try:
                    ib.cancelOrder(self.sl_trade.order)
                except Exception as exc:
                    log.warning("[%s] Cancel TP2 SL failed: %s", self.symbol, exc)
                trail               = Order()
                trail.action        = close_action
                trail.orderType     = "TRAIL"
                trail.totalQuantity = tp3_qty
                trail.auxPrice      = 1.00          # trail $1 behind market
                self.sl_trade       = ib.placeOrder(contract, trail)
                self._wire_sl_fill_log(self.sl_trade, tp3_price)
                log.info("[%s] Trailing stop ($1) live for %d shares.",
                         self.symbol, tp3_qty)

            tp2_trade.fillEvent += _on_tp2_fill

        # ── TP3 fill callback: final leg ──────────────────────────────────────
        if tp3_trade is not None:
            def _on_tp3_fill(trade, fill):
                fill_px  = fill.execution.price
                fill_qty = int(fill.execution.shares)
                log.info("[%s] TP3 filled at %.4f — closing %d shares (final leg)",
                         self.symbol, fill_px, fill_qty)
                self._log_exit(fill_px, fill_qty, "TP3", tp3_price)
                self._final_exit_logged = True

            tp3_trade.fillEvent += _on_tp3_fill

        # ── Initial SL fill callback (before any TP hits) ─────────────────────
        self._wire_sl_fill_log(sl_trade, tp1_price)

    # ── Main poll ─────────────────────────────────────────────────────────────

    async def poll(self, ib, contract) -> None:
        """Fetch fresh bars, enforce time rules, manage position lifecycle."""
        log.info("[%s] Poll — side=%+d tp1=%s tp2=%s",
                 self.symbol, self.position_side, self.tp1_hit, self.tp2_hit)

        # ── Fetch latest 15-min bars ──────────────────────────────────────────
        try:
            ib_bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="15 mins",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
        except Exception as exc:
            log.warning("[%s] reqHistoricalData error: %s", self.symbol, exc)
            return

        df = self._update_buffer(ib_bars)
        if df is None or len(df) < 50:
            log.warning("[%s] Insufficient bars (%d) — skipping.",
                        self.symbol, len(df) if df is not None else 0)
            return

        # ── Sync actual IB position ───────────────────────────────────────────
        actual_qty = next(
            (abs(int(p.position)) for p in ib.positions()
             if p.contract.symbol == self.symbol),
            0,
        )
        if actual_qty == 0 and self.position_side != 0:
            if not self._final_exit_logged:
                # IB position is gone but no TP3/SL callback fired — manual TWS close.
                if self.tp1_hit and self.tp2_hit:
                    manual_qty = self.tp3_qty
                elif self.tp1_hit:
                    manual_qty = self.tp2_qty + self.tp3_qty
                else:
                    manual_qty = self.full_qty
                last_px = float(df.iloc[-1]["close"]) if (df is not None and not df.empty) else 0.0
                log.warning(
                    "[%s] MANUAL close detected — IB position gone, no exit callback fired. "
                    "Logging MANUAL exit: qty=%d  last_bar_px=%.4f",
                    self.symbol, manual_qty, last_px,
                )
                _trade_log.log_trade(
                    symbol=self.symbol,
                    direction="LONG" if self.position_side == 1 else "SHORT",
                    entry_time=self.entry_time or datetime.now(_EST),
                    entry_price=self.entry_price or 0.0,
                    exit_time=datetime.now(_EST),
                    exit_price=last_px,
                    quantity=manual_qty,
                    strategy_name=self.strategy_name,
                    confluence_score=self.confluence_score,
                    stop_loss=self.initial_sl or 0.0,
                    take_profit=self.tp1_price or 0.0,
                    exit_reason="MANUAL",
                )
            else:
                log.info("[%s] Position fully closed (TP3 / SL / trail callback confirmed).", self.symbol)
            self._reset_position()

        # ── Time-based guards ─────────────────────────────────────────────────
        now_est                     = datetime.now(_EST)
        close_due,  close_reason    = _eod_close_due(now_est)
        entries_off, entries_reason = _entries_blocked(now_est)

        log.info(
            "[%s] Poll time: %s EST  eod_close_due=%s  entries_blocked=%s",
            self.symbol, now_est.strftime("%H:%M:%S"), close_due, entries_off,
        )
        print(
            f"[{self.symbol}] Poll {now_est.strftime('%H:%M:%S')} EST | "
            f"eod_close_due={close_due} | entries_blocked={entries_off} | "
            f"position_side={self.position_side}",
            flush=True,
        )

        if close_due:
            if self.position_side != 0:
                log.info("[%s] %s.", self.symbol, close_reason)
                self.close_position(ib, contract, close_reason)
            elif actual_qty > 0:
                # Engine state desynced — IB still holds a position; force close it.
                log.warning(
                    "[%s] EOD — engine shows flat but IB reports %d open contract(s); "
                    "force-closing now.",
                    self.symbol, actual_qty,
                )
                self._force_close_ib(ib, contract, close_reason)
            else:
                log.info("[%s] %s — already flat.", self.symbol, close_reason)
            return

        # ── In position: TP/SL callbacks handle adjustments; log state ────────
        if self.position_side != 0:
            log.info(
                "[%s] Holding side=%+d entry=%.2f sl=%.2f remaining=%d "
                "tp1_hit=%s tp2_hit=%s",
                self.symbol, self.position_side,
                self.entry_price or 0,
                self.initial_sl  or 0,
                actual_qty,
                self.tp1_hit, self.tp2_hit,
            )
            return

        # ── Entry gate ────────────────────────────────────────────────────────
        if entries_off:
            log.info("[%s] %s — skipping entry.", self.symbol, entries_reason)
            return

        # ── Run signals ───────────────────────────────────────────────────────
        try:
            signals = self.system.generate_signals(df, self.params)
        except Exception as exc:
            log.error("[%s] Signal generation error: %s", self.symbol, exc)
            return

        last     = signals.iloc[-1]
        is_long  = bool(last.get("long_signal",  False))
        is_short = bool(last.get("short_signal", False))

        if not is_long and not is_short:
            log.info("[%s] No signal on last bar.", self.symbol)
            return

        entry = float(last.get("entry_price", np.nan))
        sl    = float(last.get("stop_loss",   np.nan))

        if np.isnan(entry) or np.isnan(sl):
            log.warning("[%s] Signal prices contain NaN — skipping.", self.symbol)
            return

        # ── Size position ($10,000 fixed risk per trade) ──────────────────────
        qty = self._calc_qty(entry, sl)
        if qty <= 0:
            log.warning("[%s] qty=0 — skipping.", self.symbol)
            return

        action           = "BUY" if is_long else "SELL"
        strategy_name    = str(last.get("primary_strategy",  ""))
        confluence_score = int(last.get("confluence_count", 0))
        log.info("[%s] Signal: %s qty=%d entry=%.2f sl=%.2f strat=%s conf=%d",
                 self.symbol, action, qty, entry, sl, strategy_name, confluence_score)

        await self._enter_and_place_tps(
            ib, contract, action, qty, entry, sl,
            strategy_name=strategy_name,
            confluence_score=confluence_score,
        )


# ── Front-month contract resolution & roll ────────────────────────────────────

async def _resolve_front_month(ib, symbol: str):
    """
    Return the nearest futures contract whose expiry is more than
    _ROLL_DAYS_BEFORE_EXPIRY days away — i.e., the contract we should trade.
    """
    from datetime import date, timedelta
    from ib_insync import Future

    spec     = _CONTRACT_SPECS[symbol]
    template = Future(symbol, exchange=spec["exchange"], currency="USD")
    details  = await ib.reqContractDetailsAsync(template)

    if not details:
        raise ValueError(f"reqContractDetailsAsync returned nothing for {symbol}")

    threshold  = date.today() + timedelta(days=_ROLL_DAYS_BEFORE_EXPIRY)
    candidates = []
    for cd in details:
        exp_str = cd.contract.lastTradeDateOrContractMonth or ""
        try:
            exp_date = datetime.strptime(exp_str[:8], "%Y%m%d").date()
        except ValueError:
            continue
        if exp_date > threshold:
            candidates.append((exp_date, cd.contract))

    if not candidates:
        raise ValueError(
            f"No valid front-month contract for {symbol} "
            f"(threshold: >{threshold})"
        )

    candidates.sort(key=lambda x: x[0])
    chosen, exp = candidates[0][1], candidates[0][0]
    log.info("[%s] Front-month contract: %s (exp %s)", symbol, chosen.localSymbol or chosen.lastTradeDateOrContractMonth, exp)
    return chosen


async def _check_roll(ib, symbol: str, contract) -> object:
    """Return a new front-month contract if the current one is within the roll window."""
    if symbol not in _CONTRACT_SPECS:
        return contract

    from datetime import date
    exp_str = contract.lastTradeDateOrContractMonth or ""
    try:
        exp_date  = datetime.strptime(exp_str[:8], "%Y%m%d").date()
        days_left = (exp_date - date.today()).days
    except ValueError:
        return contract

    if days_left <= _ROLL_DAYS_BEFORE_EXPIRY:
        log.info("[%s] Rolling contract — %d day(s) to expiry (%s)",
                 symbol, days_left, exp_date)
        new_contract = await _resolve_front_month(ib, symbol)
        try:
            await ib.qualifyContractsAsync(new_contract)
        except Exception as exc:
            log.warning("[%s] Roll qualification warning: %s", symbol, exc)
        return new_contract

    return contract


# ── Orchestrator ──────────────────────────────────────────────────────────────

class IBBridge:
    """
    Manages a single IB connection and runs independent per-symbol polling loops.

    Each symbol gets its own SymbolEngine with independent position tracking,
    signal generation, and order management. Risk is 1 % of account per symbol.
    """

    POLL_INTERVAL_SECS = 60

    def __init__(
        self,
        symbols:     List[str],
        params_file: Path,
        paper:       bool = True,
    ):
        self.symbols = symbols
        self.paper   = paper
        self.params  = self._load_params(params_file)
        self.engines: Dict[str, SymbolEngine] = {
            s: SymbolEngine(s, self.params, paper) for s in symbols
        }

    @staticmethod
    def _load_params(path: Path) -> dict:
        if path and path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as exc:
                log.warning("Could not load params from %s: %s", path, exc)
        return {}

    def run(self) -> None:
        from ib_insync import util
        util.run(self._main())

    async def _main(self) -> None:
        from ib_insync import IB

        ib   = IB()
        port = IB_CFG["port_paper"] if self.paper else IB_CFG["port_live"]

        log.info(
            "Connecting to IB TWS at %s:%d (clientId=%d) [%s]…",
            IB_CFG["host"], port, IB_CFG["client_id"],
            "paper" if self.paper else "LIVE",
        )

        try:
            await ib.connectAsync(
                IB_CFG["host"],
                port,
                clientId=IB_CFG["client_id"],
                timeout=IB_CFG["timeout_secs"],
            )
        except Exception as exc:
            log.error("IB connection failed: %s", exc)
            log.error(
                "Ensure TWS/Gateway is running on port %d with API enabled.\n"
                "  TWS: File → Global Configuration → API → Settings\n"
                "  Enable: 'Enable ActiveX and Socket Clients'\n"
                "  Socket port: %d",
                port, port,
            )
            sys.exit(1)

        log.info("Connected — server=%s  accounts=%s",
                 ib.client.serverVersion(), ib.managedAccounts())

        # Resolve and qualify contracts
        contracts: Dict[str, object] = {}
        for sym, eng in self.engines.items():
            if sym in _CONTRACT_SPECS:
                # Futures: auto-select front month, respecting 5-day roll window
                contract = await _resolve_front_month(ib, sym)
            else:
                # ETFs (paper mode only)
                contract = eng.make_contract()
            try:
                await ib.qualifyContractsAsync(contract)
                log.info("[%s] Contract qualified: %s", sym, contract)
            except Exception as exc:
                log.warning("[%s] Contract qualification warning: %s", sym, exc)
            contracts[sym] = contract

        # Prime each engine with historical data
        for eng in self.engines.values():
            eng.load_warmup()

        log.info("Starting independent %ds polling loops for: %s",
                 self.POLL_INTERVAL_SECS, self.symbols)
        print("=" * 60, flush=True)
        print(f"  Trade logger ready : output/APEX_Trade_Log.xlsx", flush=True)
        print(f"  EOD close scheduled: 3:45pm EST (Mon–Thu)", flush=True)
        print(f"  Friday close       : 3:30pm EST", flush=True)
        print(f"  Symbols            : {', '.join(self.symbols)}", flush=True)
        print(f"  Poll interval      : {self.POLL_INTERVAL_SECS}s", flush=True)
        print("=" * 60, flush=True)

        tasks = [
            asyncio.create_task(
                self._run_symbol(ib, sym, contracts[sym]),
                name=f"poll-{sym}",
            )
            for sym in self.symbols
        ]

        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Shutdown signal received.")
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            ib.disconnect()
            log.info("Disconnected from IB.")

    async def _run_symbol(self, ib, sym: str, contract) -> None:
        """Independent 60-second polling loop for a single symbol."""
        while True:
            try:
                contract = await _check_roll(ib, sym, contract)
                await self.engines[sym].poll(ib, contract)
            except asyncio.CancelledError:
                log.info("[%s] Poll task cancelled.", sym)
                raise
            except Exception as exc:
                log.error("[%s] Unhandled poll error: %s", sym, exc)
            await asyncio.sleep(self.POLL_INTERVAL_SECS)
