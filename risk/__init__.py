"""
Phase 5 — Risk Management module.

Exports:
    PositionSizer     — fixed-fractional sizing with confluence scaling
    DrawdownManager   — stateful daily/weekly/peak drawdown tracker
    RiskManager       — combined interface for engine and live bridge

Quick usage:
    from risk import RiskManager
    rm = RiskManager(symbol="MES", account_size=25_000)
    allowed, contracts, reason = rm.check_entry(
        equity        = 25_000,
        entry_price   = 4500.0,
        stop_loss     = 4495.0,
        dt            = datetime.now(timezone.utc),
        direction     = 1,
        confluence_count = 4,
    )
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

from config.logger import get_logger
from config.settings import INSTRUMENTS, MIN_CONFLUENCE_CONFIRMATIONS, RISK
from risk.drawdown_manager import DrawdownManager
from risk.position_sizer   import PositionSizer

log = get_logger(__name__)


class RiskManager:
    """
    Single entry-point for all position-level risk decisions.

    check_entry() — call before opening any trade; returns (allowed, contracts, reason)
    update()      — call after each trade closes to refresh equity
    is_halted()   — quick live-loop check
    get_stats()   — full state snapshot for logging / reports
    """

    def __init__(
        self,
        symbol:             str   = "MES",
        account_size:       float = RISK["account_size"],
        risk_pct:           float = RISK["max_risk_per_trade_pct"],
        max_daily_dd:       float = RISK["max_daily_drawdown_pct"],
        max_weekly_dd:      float = RISK["max_weekly_drawdown_pct"],
        min_rr:             float = RISK["min_rr_ratio"],
        max_contracts:      int   = 500,
        min_contracts:      int   = 1,
        min_confluence:     int   = MIN_CONFLUENCE_CONFIRMATIONS,
        confluence_scaling: bool  = True,
    ):
        instrument      = INSTRUMENTS[symbol]
        self.symbol     = symbol
        self.min_rr     = min_rr
        self.min_confluence = min_confluence

        self.sizer = PositionSizer(
            point_value            = instrument["point_value"],
            tick_size              = instrument["tick_size"],
            risk_pct               = risk_pct / 100,
            min_contracts          = min_contracts,
            max_contracts          = max_contracts,
            use_confluence_scaling = confluence_scaling,
        )
        self.drawdown = DrawdownManager(
            initial_equity = account_size,
            max_daily_pct  = max_daily_dd  / 100,
            max_weekly_pct = max_weekly_dd / 100,
        )

    # ── Main decision gate ─────────────────────────────────────────────────────

    def check_entry(
        self,
        equity:           float,
        entry_price:      float,
        stop_loss:        float,
        dt:               datetime,
        direction:        int,
        confluence_count: int            = 0,
        take_profit:      Optional[float] = None,
    ) -> Tuple[bool, int, str]:
        """
        Decide whether a new entry is allowed and how many contracts to trade.

        Returns
        -------
        allowed   : bool  — True if the trade may proceed
        contracts : int   — number of contracts (0 when not allowed)
        reason    : str   — 'ok' or a short reason string when blocked
        """
        # 1. Refresh drawdown state for this timestamp
        self.drawdown.update(equity, dt)

        # 2. Drawdown halt check
        if self.drawdown.is_halted():
            return False, 0, f"drawdown_halt:{self.drawdown.halt_reason()}"

        # 3. Stop-loss on correct side
        if direction == 1  and stop_loss >= entry_price:
            return False, 0, "sl_above_entry"
        if direction == -1 and stop_loss <= entry_price:
            return False, 0, "sl_below_entry"

        sl_dist = abs(entry_price - stop_loss)

        # 4. Minimum R:R
        if take_profit is not None:
            tp_dist = abs(take_profit - entry_price)
            if sl_dist > 0 and (tp_dist / sl_dist) < self.min_rr:
                return False, 0, f"rr_too_low:{tp_dist / sl_dist:.2f}"

        # 5. Position sizing
        contracts = self.sizer.get_contracts(
            equity           = equity,
            sl_distance_pts  = sl_dist,
            confluence_count = confluence_count,
            min_confluence   = self.min_confluence,
        )
        return True, contracts, "ok"

    # ── State helpers ──────────────────────────────────────────────────────────

    def update(self, equity: float, dt: datetime) -> None:
        """Update equity after a trade closes (or each live bar)."""
        self.drawdown.update(equity, dt)

    def is_halted(self) -> bool:
        return self.drawdown.is_halted()

    def get_contracts(
        self,
        equity:           float,
        sl_distance_pts:  float,
        confluence_count: int = 0,
    ) -> int:
        """Direct sizing call without the full entry check."""
        return self.sizer.get_contracts(
            equity, sl_distance_pts, confluence_count, self.min_confluence
        )

    def reset(self, equity: float) -> None:
        """Full state reset — call at the start of each backtest."""
        self.drawdown.reset(equity)

    def get_stats(self) -> dict:
        return {"symbol": self.symbol, **self.drawdown.get_stats()}


__all__ = ["PositionSizer", "DrawdownManager", "RiskManager"]
