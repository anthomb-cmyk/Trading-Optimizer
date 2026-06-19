"""
Abstract base class for all APEX SMC strategies.

System hierarchy:
    "bias_filter"   → Strategy 10 (DailyBiasIntraday)   — sets daily direction
    "timing_filter" → Strategy 6  (KillZoneReversal)     — gates entry windows
    "primary"       → Strategies 1-3 (LS, OB, FVG)      — trigger entries
    "confirmation"  → Strategies 4,5,7,8,9               — add confluence weight

Every strategy must implement:
    generate_signals(df, params)  → df with signal columns appended
    get_param_space(trial)        → dict of params from Optuna trial
    default_params                → property returning default param dict

Signal columns added to df:
    long_signal      bool   enter long this bar
    short_signal     bool   enter short this bar
    entry_price      float  suggested entry
    stop_loss        float  stop loss price
    take_profit      float  take profit price
    confluence_count int    number of confirmations present
    valid            bool   confluence_count >= MIN_CONFLUENCE_CONFIRMATIONS
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    INSTRUMENTS,
    MIN_CONFLUENCE_CONFIRMATIONS,
    RISK,
    SMC,
    VOLATILITY,
)
from strategies.detectors import (  # causal detector functions (T1.2)
    detect_fvg,
    detect_fvg_zones,
    detect_order_blocks,
    detect_bos,
    detect_choch,
)


_SIGNAL_COLS = {
    "long_signal":      False,
    "short_signal":     False,
    "entry_price":      np.nan,
    "stop_loss":        np.nan,
    "take_profit":      np.nan,
    "confluence_count": 0,
    "valid":            False,
}


class BaseStrategy(ABC):
    """All SMC strategies inherit from this class."""

    name: str         = "base"
    system_role: str  = "primary"       # bias_filter | timing_filter | primary | confirmation
    required_timeframes: list = ["15m"]

    def __init__(self, symbol: str = "MES", min_confluence: int = MIN_CONFLUENCE_CONFIRMATIONS):
        self.symbol         = symbol
        self.instrument     = INSTRUMENTS[symbol]
        self.min_confluence = min_confluence

    # ── System-level interface ────────────────────────────────────────────────

    def get_bias(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Override in bias_filter strategies.
        Returns +1 (bullish), -1 (bearish), or 0 (neutral) per bar.
        Default falls back to EMA-200 bias so all strategies have a usable value.
        """
        return self.htf_bias(df)

    def get_timing_mask(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Override in timing_filter strategies.
        Returns True on bars where timing conditions are met.
        Default: always True (no timing restriction at base level).
        """
        return pd.Series(True, index=df.index)

    def get_confirmation(
        self,
        df: pd.DataFrame,
        direction: pd.Series,
        params: Dict[str, Any],
    ) -> pd.Series:
        """
        Used by StrategySystem to harvest confirmation signals.
        direction: +1 for long, -1 for short (per bar)
        Returns int Series: 1 where this strategy confirms, 0 otherwise.
        Default: runs generate_signals and checks alignment.
        """
        p   = {**self.default_params, **params}
        sig = self.generate_signals(df, p)
        confirms = (
            (direction == 1)  & sig["long_signal"] |
            (direction == -1) & sig["short_signal"]
        )
        return confirms.astype(int)

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Return df with signal columns added. No look-ahead past current bar."""

    @abstractmethod
    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        """Return a dict of parameters sampled from an Optuna trial."""

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Default parameter values used when not optimizing."""

    # ── Signal column helpers ────────────────────────────────────────────────

    @staticmethod
    def _init_signal_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, default in _SIGNAL_COLS.items():
            df[col] = default
        return df

    @staticmethod
    def _finalize_signals(df: pd.DataFrame, min_confluence: int) -> pd.DataFrame:
        df["valid"]        = df["confluence_count"] >= min_confluence
        df["long_signal"]  = df["long_signal"]  & df["valid"]
        df["short_signal"] = df["short_signal"] & df["valid"]
        return df

    # ── SMC detectors — implementations live in strategies/detectors.py (T1.2) ──

    @staticmethod
    def detect_fvg(
        df: pd.DataFrame,
        min_atr_mult: float = 0.3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Fair Value Gap detection. Causal as-is. See strategies/detectors.py."""
        return detect_fvg(df, min_atr_mult)

    @staticmethod
    def detect_fvg_zones(
        df: pd.DataFrame,
        min_atr_mult: float = 0.3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """FVG zone bounds. Causal as-is. See strategies/detectors.py."""
        return detect_fvg_zones(df, min_atr_mult)

    @staticmethod
    def detect_order_blocks(
        df: pd.DataFrame,
        lookback: int = 20,
        body_pct: float = 0.60,
        displacement_atr_mult: float = 0.8,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Order Block detection via displacement confirmation.

        CAUSALITY FIX (T1.2): flag is now placed at the DISPLACEMENT bar
        (i+j), not the OB candle (i).  This is the first bar at which the
        OB's existence is knowable from past data only.  Entries fire on
        retest (bars > i+j), so trading semantics are unchanged.

        Returns a 6-tuple:
            (bull_ob, bear_ob, bull_ob_high, bull_ob_low, bear_ob_high, bear_ob_low)

        The _high/_low Series carry the OB candle's high/low to the
        displacement (confirmation) bar; NaN elsewhere.  Consumers .ffill()
        them to propagate the zone forward.  Causality is preserved.

        See strategies/detectors.py for the full implementation.
        """
        return detect_order_blocks(df, lookback, body_pct, displacement_atr_mult)

    @staticmethod
    def detect_bos(
        df: pd.DataFrame,
        lookback: int = 20,
        confirm_bars: int = 1,
    ) -> Tuple[pd.Series, pd.Series]:
        """Break of Structure. Causal as-is. See strategies/detectors.py."""
        return detect_bos(df, lookback, confirm_bars)

    @staticmethod
    def detect_choch(
        df: pd.DataFrame,
        lookback: int = 20,
    ) -> Tuple[pd.Series, pd.Series]:
        """Change of Character. Causal as-is. See strategies/detectors.py."""
        return detect_choch(df, lookback)

    # ── Shared filter utilities ───────────────────────────────────────────────

    @staticmethod
    def htf_bias(df: pd.DataFrame, ema_col: str = "ema_200") -> pd.Series:
        """+1 bullish, -1 bearish, 0 neutral based on price vs EMA."""
        if ema_col not in df.columns:
            return pd.Series(0, index=df.index)
        bias = np.where(
            df["close"] > df[ema_col],  1,
            np.where(df["close"] < df[ema_col], -1, 0)
        )
        return pd.Series(bias, index=df.index)

    @staticmethod
    def kill_zone_mask(
        df: pd.DataFrame,
        zones: list = ("london", "ny_am"),
    ) -> pd.Series:
        """True for bars inside any of the specified kill zones (UTC)."""
        from config.settings import KILL_ZONES

        def _tf(t: str) -> float:
            h, m = t.split(":")
            return int(h) + int(m) / 60.0

        hour = df.index.hour + df.index.minute / 60.0
        mask = pd.Series(False, index=df.index)
        for zone in zones:
            kz = KILL_ZONES[zone]
            mask |= (hour >= _tf(kz["start"])) & (hour < _tf(kz["end"]))
        return mask

    @staticmethod
    def volatility_filter(
        df: pd.DataFrame,
        max_mult: float = 3.0,
        min_mult: float = 0.3,
    ) -> pd.Series:
        """True where ATR is within acceptable bounds."""
        atr     = df["atr"].ffill()
        atr_avg = df["atr_avg"].ffill()
        ok = (atr <= atr_avg * max_mult) & (atr >= atr_avg * min_mult)
        return ok.fillna(True)

    @staticmethod
    def volume_filter(df: pd.DataFrame, min_mult: float = 1.0) -> pd.Series:
        """True where volume >= min_mult * rolling average."""
        if "vol_ma" not in df.columns or df["vol_ma"].isna().all():
            return pd.Series(True, index=df.index)
        return (df["volume"] >= df["vol_ma"] * min_mult).fillna(True)

    @staticmethod
    def rsi_filter(
        df: pd.DataFrame,
        direction: int,
        ob_level: float = 70,
        os_level: float = 30,
    ) -> pd.Series:
        if "rsi" not in df.columns:
            return pd.Series(True, index=df.index)
        if direction == 1:
            return df["rsi"] < ob_level
        return df["rsi"] > os_level

    @staticmethod
    def fibonacci_levels(swing_low: float, swing_high: float) -> Dict[str, float]:
        diff = swing_high - swing_low
        return {
            "0.0":   swing_high,
            "0.236": swing_high - 0.236 * diff,
            "0.382": swing_high - 0.382 * diff,
            "0.5":   swing_high - 0.500 * diff,
            "0.618": swing_high - 0.618 * diff,
            "0.705": swing_high - 0.705 * diff,
            "0.786": swing_high - 0.786 * diff,
            "1.0":   swing_low,
            "1.272": swing_low  - 0.272 * diff,
            "1.618": swing_low  - 0.618 * diff,
        }

    # ── Common helper: rolling any ────────────────────────────────────────────

    @staticmethod
    def _rolling_any(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=1).max().astype(bool)

    @staticmethod
    def _bars_since(s: pd.Series) -> pd.Series:
        """Bars elapsed since last True value; 9999 if never seen."""
        arr    = s.values.astype(bool)
        result = np.full(len(arr), 9999, dtype=np.int32)
        last   = -9999
        for i, v in enumerate(arr):
            if v:
                last = i
            result[i] = i - last if last >= 0 else 9999
        return pd.Series(result, index=s.index)
