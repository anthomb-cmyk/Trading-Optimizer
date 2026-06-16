"""
Volatility Circuit Breaker — blocks new entries when market conditions fall
outside acceptable ATR bounds.

Two checks applied per bar:
  1. Spike guard  : ATR > atr_max_mult × avg_ATR  → skip (too dangerous)
  2. Low-vol guard: ATR < atr_min_mult × avg_ATR  → skip (no edge)

Day-halt mode (default ON):
    Once a spike bar is detected, every remaining bar on that UTC calendar
    day is also blocked.  Resets at midnight UTC.  This prevents entering
    into chaotic conditions that persist after an initial spike.

Single-bar check (check_bar) is provided for live trading.
"""
import pandas as pd

from config.logger import get_logger
from config.settings import VOLATILITY

log = get_logger(__name__)


class VolatilityCircuitBreaker:

    def __init__(
        self,
        atr_max_mult: float = VOLATILITY["atr_multiplier_max"],
        atr_min_mult: float = VOLATILITY["low_volatility_filter"],
        day_halt:     bool  = True,
    ):
        self.atr_max_mult = atr_max_mult
        self.atr_min_mult = atr_min_mult
        self.day_halt     = day_halt

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Boolean Series (True = OK to trade).

        With day_halt=True, a spike on any bar of a calendar day halts all
        subsequent bars for that day.
        """
        atr     = df["atr"].ffill()
        atr_avg = df["atr_avg"].ffill()

        spike   = (atr > atr_avg * self.atr_max_mult).fillna(False)
        low_vol = (atr < atr_avg * self.atr_min_mult).fillna(False)
        bar_ok  = ~spike & ~low_vol

        if not self.day_halt:
            return bar_ok.rename("vol_ok")

        # Propagate spike halt to rest of the UTC calendar day
        date_labels = pd.Series(df.index.date, index=df.index)
        halted      = pd.Series(False, index=df.index)

        for date, group_idx in date_labels.groupby(date_labels).groups.items():
            day_spike = spike.loc[group_idx]
            if day_spike.any():
                first_spike_pos       = int(day_spike.values.argmax())
                halt_idx              = group_idx[first_spike_pos:]
                halted.loc[halt_idx]  = True

        ok = (~halted & bar_ok).rename("vol_ok")

        halted_n = int((~ok).sum())
        if halted_n:
            log.debug(
                "VolatilityCircuitBreaker: %d bars blocked (%.1f%%).",
                halted_n, 100 * halted_n / max(len(df), 1),
            )
        return ok

    def check_bar(self, atr: float, atr_avg: float) -> bool:
        """Single-bar check for live trading.  True = OK to trade."""
        if atr_avg <= 0:
            return True
        if atr > atr_avg * self.atr_max_mult:
            return False
        if atr < atr_avg * self.atr_min_mult:
            return False
        return True
