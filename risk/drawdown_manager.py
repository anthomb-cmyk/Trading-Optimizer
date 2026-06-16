"""
Stateful drawdown tracker for daily, weekly, and all-time peak drawdown.

Halt logic:
    daily  DD >= max_daily_pct   → halt until start of next UTC day
    weekly DD >= max_weekly_pct  → halt until start of next UTC Monday

Anchors reset automatically when a new period is detected via update().
Peak drawdown is a running max — it never resets.

Designed to be shared between BacktestEngine and the Phase-6 live bridge,
so all state lives here (not scattered across the call sites).
"""
from datetime import date, datetime
from typing import Optional

from config.logger import get_logger
from config.settings import RISK

log = get_logger(__name__)


class DrawdownManager:

    def __init__(
        self,
        initial_equity: float,
        max_daily_pct:  float = RISK["max_daily_drawdown_pct"]  / 100,
        max_weekly_pct: float = RISK["max_weekly_drawdown_pct"] / 100,
    ):
        self.max_daily_pct  = max_daily_pct
        self.max_weekly_pct = max_weekly_pct
        self._init(initial_equity)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, equity: float, dt) -> None:
        """
        Feed the current equity at a given bar/trade timestamp.
        Resets period anchors automatically on day/week transitions.
        """
        self._maybe_reset_periods(dt)
        self._equity = equity

        if equity > self._peak:
            self._peak = equity

        if self._peak > 0:
            self._peak_dd = max(self._peak_dd, (self._peak - equity) / self._peak)

        self._evaluate_halt()

    def is_halted(self) -> bool:
        """True = do not open new trades."""
        return self._halt_reason is not None

    def halt_reason(self) -> Optional[str]:
        """'daily' | 'weekly' | None"""
        return self._halt_reason

    def clear_halt(self) -> None:
        """Manually clear halt (used in unit tests; normally cleared by period reset)."""
        self._halt_reason = None

    @property
    def daily_drawdown(self) -> float:
        """Current day's loss from the daily anchor (0.0–1.0)."""
        if self._daily_anchor <= 0:
            return 0.0
        return max(0.0, (self._daily_anchor - self._equity) / self._daily_anchor)

    @property
    def weekly_drawdown(self) -> float:
        """Current week's loss from the weekly anchor (0.0–1.0)."""
        if self._weekly_anchor <= 0:
            return 0.0
        return max(0.0, (self._weekly_anchor - self._equity) / self._weekly_anchor)

    @property
    def peak_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown ever seen (0.0–1.0)."""
        return self._peak_dd

    def get_stats(self) -> dict:
        return {
            "equity":          round(self._equity, 2),
            "peak_equity":     round(self._peak, 2),
            "daily_drawdown":  round(self.daily_drawdown, 4),
            "weekly_drawdown": round(self.weekly_drawdown, 4),
            "peak_drawdown":   round(self._peak_dd, 4),
            "halted":          self.is_halted(),
            "halt_reason":     self._halt_reason,
        }

    def reset(self, equity: float) -> None:
        """Full reset — call at the start of each backtest run."""
        self._init(equity)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _init(self, equity: float) -> None:
        self._equity        = equity
        self._peak          = equity
        self._daily_anchor  = equity
        self._weekly_anchor = equity
        self._peak_dd       = 0.0
        self._halt_reason: Optional[str] = None
        self._last_date: Optional[date]  = None
        self._last_week: Optional[int]   = None

    def _maybe_reset_periods(self, dt) -> None:
        d = dt.date() if hasattr(dt, "date") and callable(dt.date) else dt
        if isinstance(d, datetime):
            d = d.date()
        try:
            w = d.isocalendar()[1]
        except AttributeError:
            return

        if self._last_date is None or d > self._last_date:
            self._daily_anchor = self._equity
            if self._halt_reason == "daily":
                self._halt_reason = None
            self._last_date = d

        if self._last_week is None or w != self._last_week:
            self._weekly_anchor = self._equity
            if self._halt_reason == "weekly":
                self._halt_reason = None
            self._last_week = w

    def _evaluate_halt(self) -> None:
        # Weekly supersedes daily
        if self.weekly_drawdown >= self.max_weekly_pct:
            if self._halt_reason != "weekly":
                log.warning(
                    "Weekly DD limit reached: %.2f%% (limit %.2f%%)",
                    self.weekly_drawdown * 100, self.max_weekly_pct * 100,
                )
                self._halt_reason = "weekly"
            return

        if self.daily_drawdown >= self.max_daily_pct:
            if self._halt_reason != "daily":
                log.warning(
                    "Daily DD limit reached: %.2f%% (limit %.2f%%)",
                    self.daily_drawdown * 100, self.max_daily_pct * 100,
                )
                self._halt_reason = "daily"
