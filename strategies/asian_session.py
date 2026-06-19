"""
Strategy 7 — Asian Session Range Breakout

Logic:
  The Asian session (00:00–09:00 UTC) tends to be low-volume range-building.
  London and NY sessions frequently seek liquidity at the Asian session's
  high or low before making a directional move.

  Two setups:
    A) Breakout: Price breaks the Asian range during London/NY → enter on retest.
    B) Fade:     Price sweeps Asian high/low then snaps back into range → fade it.

  Default is Breakout mode (more reliable with trend).

Confluence checklist:
  [1] Asian session range defined (not too wide, not too tight)     ← required
  [2] Break of Asian high or low confirmed (close outside range)
  [3] Retest of broken level (entry trigger)
  [4] HTF bias aligned with breakout direction
  [5] Inside London or NY session
  [6] Volume expansion on breakout candle
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.settings import KILL_ZONES
from strategies.base_strategy import BaseStrategy


class AsianSessionRangeBreakout(BaseStrategy):
    name        = "asian_session_range_breakout"
    system_role = "confirmation"
    required_timeframes = ["15m", "5m"]

    # UTC hours for Asian session
    _ASIAN_START = 0
    _ASIAN_END   = 9

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "min_range_atr":      0.5,     # Asian range must be >= x * ATR
            "max_range_atr":      4.0,     # Asian range must be <= x * ATR (avoid wild days)
            "retest_tolerance_atr": 0.3,
            "breakout_confirm_bars": 1,    # bars to confirm close outside range
            "mode":               "breakout",  # 'breakout' | 'fade'
            "atr_sl_mult":        1.0,
            "rr_ratio":           2.0,
            "active_sessions":    ["london", "ny_am"],
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "min_range_atr":         trial.suggest_float("asr_min_range_atr",       0.3, 1.5),
            "max_range_atr":         trial.suggest_float("asr_max_range_atr",       2.0, 6.0),
            "retest_tolerance_atr":  trial.suggest_float("asr_retest_tol_atr",      0.1, 0.6),
            "breakout_confirm_bars": trial.suggest_int("asr_confirm_bars",          1, 4),
            "mode":                  trial.suggest_categorical("asr_mode", ["breakout", "fade"]),
            "atr_sl_mult":           trial.suggest_float("asr_atr_sl_mult",         0.5, 2.0),
            "rr_ratio":              trial.suggest_float("asr_rr_ratio",            1.5, 4.0),
        }

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr  = df["atr"].ffill()
        tol  = atr * p["retest_tolerance_atr"]
        mode = p["mode"]

        # ── Step 1: Build daily Asian session high/low ─────────────────────────
        asian_high, asian_low = self._calc_asian_range(df)
        asian_size = asian_high - asian_low

        # Valid range filter
        range_ok = (
            (asian_size >= atr * p["min_range_atr"]) &
            (asian_size <= atr * p["max_range_atr"])
        )

        # ── Step 2: Active session filter ────────────────────────────────────
        hour = df.index.hour + df.index.minute / 60.0
        in_active = pd.Series(False, index=df.index)
        for sess in p.get("active_sessions", ["london", "ny_am"]):
            kz = KILL_ZONES[sess]
            def _tf(t):
                h, m = t.split(":")
                return int(h) + int(m) / 60.0
            in_active |= (hour >= _tf(kz["start"])) & (hour < _tf(kz["end"]))

        # Not in Asian session (only trade breakouts after Asian session closes)
        not_asian = (hour >= self._ASIAN_END) | (hour < self._ASIAN_START)

        # ── Step 3: Breakout detection ────────────────────────────────────────
        bull_break = (df["close"] > asian_high) & range_ok & in_active & not_asian
        bear_break = (df["close"] < asian_low)  & range_ok & in_active & not_asian

        # Retest: price comes back to range boundary
        at_asian_high = (
            (df["low"]  <= asian_high + tol) &
            (df["high"] >= asian_high - tol)
        )
        at_asian_low = (
            (df["high"] >= asian_low  - tol) &
            (df["low"]  <= asian_low  + tol)
        )

        if mode == "breakout":
            # Enter on first retest after breakout
            bull_break_recent = self._rolling_any(bull_break, 10)
            bear_break_recent = self._rolling_any(bear_break, 10)
            long_trigger  = at_asian_high & bull_break_recent & ~bull_break & in_active
            short_trigger = at_asian_low  & bear_break_recent & ~bear_break & in_active
        else:
            # Fade: sweep beyond range then close back inside → fade
            bull_fade = (
                (df["low"] < asian_low) &
                (df["close"] > asian_low) &
                range_ok & in_active & not_asian
            )
            bear_fade = (
                (df["high"] > asian_high) &
                (df["close"] < asian_high) &
                range_ok & in_active & not_asian
            )
            long_trigger  = bull_fade
            short_trigger = bear_fade

        # ── Step 4: Confluence ────────────────────────────────────────────────
        bias  = self.htf_bias(df)
        vol   = self.volume_filter(df, 1.0)

        bull_fvg, bear_fvg = self.detect_fvg(df)
        fvg_bull_w = self._rolling_any(bull_fvg, 8)
        fvg_bear_w = self._rolling_any(bear_fvg, 8)

        long_conf = (
            range_ok.astype(int) +              # [1] valid range
            (bull_break_recent.astype(int) if mode == "breakout" else long_trigger.astype(int)) +  # [2] break / trigger
            (bias == 1).astype(int) +           # [3] HTF bullish
            in_active.astype(int) +             # [4] in active session
            vol.astype(int) +                   # [5] volume
            fvg_bull_w.astype(int)              # [6] FVG
        )
        short_conf = (
            range_ok.astype(int) +
            (bear_break_recent.astype(int) if mode == "breakout" else short_trigger.astype(int)) +  # [2] break / trigger
            (bias == -1).astype(int) +
            in_active.astype(int) +
            vol.astype(int) +
            fvg_bear_w.astype(int)
        )

        out["long_signal"]  = long_trigger  & (long_conf  >= self.min_confluence)
        out["short_signal"] = short_trigger & (short_conf >= self.min_confluence)

        out.loc[out["long_signal"],  "entry_price"] = df.loc[out["long_signal"],  "close"]
        out.loc[out["short_signal"], "entry_price"] = df.loc[out["short_signal"], "close"]

        atr_sl = atr * p["atr_sl_mult"]
        out.loc[out["long_signal"],  "stop_loss"] = asian_low.loc[out["long_signal"]]  - atr_sl.loc[out["long_signal"]]
        out.loc[out["short_signal"], "stop_loss"] = asian_high.loc[out["short_signal"]] + atr_sl.loc[out["short_signal"]]

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[out["long_signal"],  "take_profit"] = out.loc[out["long_signal"],  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[out["short_signal"], "take_profit"] = out.loc[out["short_signal"], "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(out["long_signal"],  long_conf,
                                  np.where(out["short_signal"], short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _calc_asian_range(self, df: pd.DataFrame):
        """
        For each bar, look back to find the high/low of the most recent
        complete Asian session (00:00–09:00 UTC).
        Forward-fill so bars outside Asian session carry the last complete range.
        """
        hour = df.index.hour
        in_asian = (hour >= self._ASIAN_START) & (hour < self._ASIAN_END)

        asian_high = df["high"].where(in_asian).groupby(df.index.date).transform("max")
        asian_low  = df["low"].where(in_asian).groupby(df.index.date).transform("min")

        # Forward-fill to non-Asian bars
        asian_high = asian_high.ffill()
        asian_low  = asian_low.ffill()

        return asian_high.fillna(df["high"]), asian_low.fillna(df["low"])

    @staticmethod
    def _rolling_any(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=1).max().astype(bool)
