"""
Phase 4 — Filters module.

Exports:
    NewsFilter             — economic calendar blackout mask
    VolatilityCircuitBreaker — ATR-based halt mask
    FilterPipeline         — combines both; single apply() call for the engine

Usage in BacktestEngine:
    from filters import FilterPipeline
    fp = FilterPipeline()
    fp.load()
    engine = BacktestEngine(symbol, filters=fp)
    result = engine.run(df, signals, params)

Usage in live trading:
    fp.is_blocked(df_row, dt=datetime.now(timezone.utc))
"""
from __future__ import annotations

import pandas as pd

from config.logger import get_logger
from filters.news_filter       import NewsFilter
from filters.volatility_filter import VolatilityCircuitBreaker

log = get_logger(__name__)

_GATE_COLS = ("long_signal", "short_signal", "valid")


class FilterPipeline:
    """
    Combines NewsFilter + VolatilityCircuitBreaker into one apply() call.

    Signals that fall inside a blocked bar have long_signal / short_signal /
    valid set to False.  All other signal columns are left untouched.
    """

    def __init__(
        self,
        use_news:       bool = True,
        use_volatility: bool = True,
        news_kwargs:    dict = None,
        vol_kwargs:     dict = None,
    ):
        self.use_news       = use_news
        self.use_volatility = use_volatility
        self._news   = NewsFilter(**(news_kwargs or {}))          if use_news       else None
        self._vol    = VolatilityCircuitBreaker(**(vol_kwargs or {})) if use_volatility else None
        self._loaded = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self, force_news_refresh: bool = False) -> None:
        """Load events / warm up filters.  Call once before apply()."""
        if self.use_news and self._news:
            self._news.load_events(force_refresh=force_news_refresh)
        self._loaded = True

    # ── Batch use (backtesting / optimizer) ───────────────────────────────────

    def apply(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Gate signals with all active filters.
        Returns a copy of signals with blocked-bar entries zeroed out.
        """
        if not self._loaded:
            self.load()

        blocked = pd.Series(False, index=df.index)

        if self.use_news and self._news:
            blocked |= self._news.get_blackout_mask(df)

        if self.use_volatility and self._vol:
            blocked |= ~self._vol.get_mask(df)

        if not blocked.any():
            return signals

        # Align in case df and signals have slightly different indices
        blocked = blocked.reindex(signals.index, fill_value=False)
        out = signals.copy()
        for col in _GATE_COLS:
            if col in out.columns:
                out.loc[blocked, col] = False

        n = int(blocked.sum())
        log.debug(
            "FilterPipeline blocked %d / %d bars (%.1f%%).",
            n, len(signals), 100 * n / max(len(signals), 1),
        )
        return out

    # ── Single-bar use (live trading) ─────────────────────────────────────────

    def is_blocked(self, df_row: pd.Series, dt=None) -> bool:
        """
        True = this bar should NOT generate a new entry.

        df_row : a single row from the market data DataFrame
                 (needs 'atr' and 'atr_avg' for volatility check)
        dt     : timezone-aware datetime of the bar (for news check)
        """
        if not self._loaded:
            self.load()

        if self.use_news and self._news and dt is not None:
            if self._news.is_blackout(dt):
                return True

        if self.use_volatility and self._vol:
            if not self._vol.check_bar(
                float(df_row.get("atr", 0)),
                float(df_row.get("atr_avg", 0)),
            ):
                return True

        return False


__all__ = [
    "NewsFilter",
    "VolatilityCircuitBreaker",
    "FilterPipeline",
]
