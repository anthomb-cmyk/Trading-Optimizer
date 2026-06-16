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

    # ── SMC detectors (shared across all strategies) ─────────────────────────

    @staticmethod
    def detect_fvg(
        df: pd.DataFrame,
        min_atr_mult: float = 0.3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Fair Value Gap detection (3-candle imbalance).

        Bullish FVG: candle[i-2].high < candle[i].low  — unfilled gap above
        Bearish FVG: candle[i-2].low  > candle[i].high — unfilled gap below

        Returns (bull_fvg, bear_fvg) boolean Series marking the MIDDLE candle.
        """
        atr     = df["atr"].ffill()
        min_gap = atr * min_atr_mult

        high_2 = df["high"].shift(2)
        low_2  = df["low"].shift(2)

        bull_fvg = (high_2 < df["low"])  & ((df["low"]  - high_2) >= min_gap)
        bear_fvg = (low_2  > df["high"]) & ((low_2 - df["high"])  >= min_gap)

        return bull_fvg.fillna(False), bear_fvg.fillna(False)

    @staticmethod
    def detect_fvg_zones(
        df: pd.DataFrame,
        min_atr_mult: float = 0.3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns DataFrames for open (unfilled) FVG zones.
        Each row: top, bottom, midpoint of the gap, age (bars since formation).

        Bullish FVG zone: bottom = candle[i-2].high, top = candle[i].low
        Bearish FVG zone: top = candle[i-2].low, bottom = candle[i].high
        """
        atr     = df["atr"].ffill()
        min_gap = atr * min_atr_mult
        high_2  = df["high"].shift(2)
        low_2   = df["low"].shift(2)

        bull_mask = (high_2 < df["low"])  & ((df["low"]  - high_2) >= min_gap)
        bear_mask = (low_2  > df["high"]) & ((low_2 - df["high"])  >= min_gap)

        # Bullish FVG zone columns
        bull_zones = pd.DataFrame(index=df.index)
        bull_zones["top"]    = df["low"].where(bull_mask)
        bull_zones["bottom"] = high_2.where(bull_mask)
        bull_zones["mid"]    = ((bull_zones["top"] + bull_zones["bottom"]) / 2)

        bear_zones = pd.DataFrame(index=df.index)
        bear_zones["top"]    = low_2.where(bear_mask)
        bear_zones["bottom"] = df["high"].where(bear_mask)
        bear_zones["mid"]    = ((bear_zones["top"] + bear_zones["bottom"]) / 2)

        return bull_zones, bear_zones

    @staticmethod
    def detect_order_blocks(
        df: pd.DataFrame,
        lookback: int = 20,
        body_pct: float = 0.60,
        displacement_atr_mult: float = 0.8,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Order Block detection via displacement confirmation.

        Bullish OB: last bearish candle immediately before a bullish impulse
                    (the impulse bar has strong body AND range >= displacement_atr_mult * ATR)
        Bearish OB: last bullish candle immediately before a bearish impulse

        NOTE: The OB candle is marked AFTER the displacement confirms it — this is
        intentional. Entries only fire on retest (future bars), so no look-ahead
        is introduced in actual signal generation.

        Returns (bull_ob, bear_ob) boolean Series marking the OB candle.
        """
        atr        = df["atr"].ffill()
        body_ratio = df["body"] / df["range"].replace(0, np.nan)

        # Displacement: strong body + large range
        strong_bull = df["bullish"]  & (body_ratio >= body_pct) & (df["range"] >= atr * displacement_atr_mult)
        strong_bear = ~df["bullish"] & (body_ratio >= body_pct) & (df["range"] >= atr * displacement_atr_mult)

        bull_ob = pd.Series(False, index=df.index)
        bear_ob = pd.Series(False, index=df.index)

        # For each displacement, mark the last opposing candle before it
        # shift(-j) = "j bars later had a displacement" → mark current bar as OB candidate
        for j in range(1, min(lookback + 1, len(df))):
            fwd_strong_bull = strong_bull.shift(-j).fillna(False)
            fwd_strong_bear = strong_bear.shift(-j).fillna(False)
            bull_ob |= ~df["bullish"] & fwd_strong_bull
            bear_ob |= df["bullish"]  & fwd_strong_bear

        return bull_ob, bear_ob

    @staticmethod
    def detect_bos(
        df: pd.DataFrame,
        lookback: int = 20,
        confirm_bars: int = 1,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Break of Structure.
        Bullish BOS: close above the highest high of the previous lookback bars.
        Bearish BOS: close below the lowest low.
        """
        rolling_high = df["high"].rolling(lookback).max().shift(confirm_bars)
        rolling_low  = df["low"].rolling(lookback).min().shift(confirm_bars)
        bull_bos = (df["close"] > rolling_high).fillna(False)
        bear_bos = (df["close"] < rolling_low).fillna(False)
        return bull_bos, bear_bos

    @staticmethod
    def detect_choch(
        df: pd.DataFrame,
        lookback: int = 20,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Change of Character: first close through the most recent internal swing.

        Bullish CHoCH: in a downtrend, closes above the last swing high.
        Bearish CHoCH: in an uptrend, closes below the last swing low.
        """
        last_sh = df["high"].where(df["swing_high"]).ffill()
        last_sl = df["low"].where(df["swing_low"]).ffill()

        bull_choch = (
            (df["close"] > last_sh.shift(1)) &
            (df["close"].shift(1) <= last_sh.shift(1))
        ).fillna(False)
        bear_choch = (
            (df["close"] < last_sl.shift(1)) &
            (df["close"].shift(1) >= last_sl.shift(1))
        ).fillna(False)

        return bull_choch, bear_choch

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
