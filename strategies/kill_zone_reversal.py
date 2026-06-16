"""
Strategy 8 — Kill Zone Reversal

Logic:
  ICT kill zones are time windows when institutional order flow is highest.
  The market frequently sweeps liquidity (previous highs/lows, equal levels)
  at the START of a kill zone, then reverses sharply.

  Steps:
    1. At the start of a kill zone, identify the nearest liquidity pools
       (recent swing highs/lows from the PREVIOUS session).
    2. Watch for a sweep of that liquidity during the kill zone window.
    3. Look for a displacement candle (strong, large body) after the sweep
       confirming the reversal — this is the key tell.
    4. Enter on the FVG created by the displacement, or on the next bar.

Confluence checklist:
  [1] Inside a kill zone window                                  ← required
  [2] Liquidity sweep confirmed (wick beyond previous session H/L)
  [3] Displacement candle (body >= displacement_atr_mult * ATR)
  [4] FVG formed by displacement candle
  [5] HTF bias aligned with reversal direction
  [6] RSI at extreme reading at time of sweep (>70 or <30)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.settings import KILL_ZONES
from strategies.base_strategy import BaseStrategy


class KillZoneReversal(BaseStrategy):
    name        = "kill_zone_reversal"
    system_role = "timing_filter"
    required_timeframes = ["5m", "15m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "kill_zone":              "ny_am",   # 'london' | 'ny_am'
            "liq_lookback_bars":      20,         # bars before KZ for liquidity levels
            "sweep_wick_atr_mult":    0.3,        # minimum wick beyond level
            "displacement_atr_mult":  1.0,
            "displacement_body_pct":  0.6,
            "fvg_min_atr_mult":       0.3,
            "rsi_extreme_long":       35,
            "rsi_extreme_short":      65,
            "atr_sl_mult":            1.0,
            "rr_ratio":               2.5,
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "kill_zone":             trial.suggest_categorical("kzr_kill_zone", ["london", "ny_am"]),
            "liq_lookback_bars":     trial.suggest_int("kzr_liq_lookback",      10, 40),
            "sweep_wick_atr_mult":   trial.suggest_float("kzr_sweep_wick",      0.1, 0.8),
            "displacement_atr_mult": trial.suggest_float("kzr_displace_atr",    0.5, 2.0),
            "displacement_body_pct": trial.suggest_float("kzr_displace_body",   0.45, 0.80),
            "fvg_min_atr_mult":      trial.suggest_float("kzr_fvg_min_atr",     0.2, 0.6),
            "rsi_extreme_long":      trial.suggest_int("kzr_rsi_ext_long",      20, 40),
            "rsi_extreme_short":     trial.suggest_int("kzr_rsi_ext_short",     60, 80),
            "atr_sl_mult":           trial.suggest_float("kzr_atr_sl_mult",     0.6, 2.0),
            "rr_ratio":              trial.suggest_float("kzr_rr_ratio",         1.5, 4.0),
        }

    def get_timing_mask(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        System interface: returns True on bars inside the configured kill zone.
        Called by StrategySystem as the Layer-1 timing gate.
        """
        p = {**self.default_params, **params}
        return self.kill_zone_mask(df, [p["kill_zone"]])

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr = df["atr"].ffill()
        lb  = p["liq_lookback_bars"]

        # ── Step 1: Kill zone mask ────────────────────────────────────────────
        in_kz = self.kill_zone_mask(df, [p["kill_zone"]])

        # ── Step 2: Identify liquidity levels before the kill zone ───────────
        # Recent swing highs/lows from the previous lb bars (outside kill zone)
        prev_high = df["high"].rolling(lb).max().shift(1)
        prev_low  = df["low"].rolling(lb).min().shift(1)

        # ── Step 3: Sweep detection ───────────────────────────────────────────
        min_wick = atr * p["sweep_wick_atr_mult"]

        # Bullish sweep: wick below prev low and closes back above
        bull_sweep = (
            in_kz &
            (df["low"]   < prev_low  - min_wick) &
            (df["close"] > prev_low)
        )
        # Bearish sweep: wick above prev high and closes back below
        bear_sweep = (
            in_kz &
            (df["high"]  > prev_high + min_wick) &
            (df["close"] < prev_high)
        )

        # ── Step 4: Displacement candle after sweep ───────────────────────────
        displace_size = atr * p["displacement_atr_mult"]
        body_pct      = df["body"] / df["range"].replace(0, np.nan)
        strong_candle = (df["range"] >= displace_size) & (body_pct >= p["displacement_body_pct"])

        # Displacement in correct direction after sweep
        displace_bull = strong_candle & df["bullish"]  & self._rolling_any(bull_sweep, 3)
        displace_bear = strong_candle & ~df["bullish"] & self._rolling_any(bear_sweep, 3)

        # ── Step 5: FVG entry ─────────────────────────────────────────────────
        bull_fvg, bear_fvg = self.detect_fvg(df, p["fvg_min_atr_mult"])
        fvg_bull_w = self._rolling_any(bull_fvg, 5)
        fvg_bear_w = self._rolling_any(bear_fvg, 5)

        # ── Confluence ────────────────────────────────────────────────────────
        bias  = self.htf_bias(df)
        rsi   = df.get("rsi", pd.Series(50, index=df.index)).fillna(50)
        rsi_long  = rsi <= p["rsi_extreme_long"]
        rsi_short = rsi >= p["rsi_extreme_short"]

        # Entry: displacement bar or first bar after displacement
        bull_displace_recent = self._rolling_any(displace_bull, 2)
        bear_displace_recent = self._rolling_any(displace_bear, 2)

        long_conf = (
            in_kz.astype(int) +                 # [1] in kill zone
            self._rolling_any(bull_sweep, 5).astype(int) +  # [2] sweep
            bull_displace_recent.astype(int) +  # [3] displacement
            fvg_bull_w.astype(int) +            # [4] FVG
            (bias == 1).astype(int) +           # [5] HTF bullish
            rsi_long.astype(int)                # [6] RSI extreme
        )
        short_conf = (
            in_kz.astype(int) +
            self._rolling_any(bear_sweep, 5).astype(int) +
            bear_displace_recent.astype(int) +
            fvg_bear_w.astype(int) +
            (bias == -1).astype(int) +
            rsi_short.astype(int)
        )

        long_trigger  = bull_displace_recent & in_kz & (long_conf  >= self.min_confluence)
        short_trigger = bear_displace_recent & in_kz & (short_conf >= self.min_confluence)

        out["long_signal"]  = long_trigger
        out["short_signal"] = short_trigger

        out.loc[long_trigger,  "entry_price"] = df.loc[long_trigger,  "close"]
        out.loc[short_trigger, "entry_price"] = df.loc[short_trigger, "close"]

        atr_sl = atr * p["atr_sl_mult"]
        out.loc[long_trigger,  "stop_loss"] = prev_low.loc[long_trigger]   - atr_sl.loc[long_trigger]
        out.loc[short_trigger, "stop_loss"] = prev_high.loc[short_trigger] + atr_sl.loc[short_trigger]

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[long_trigger,  "take_profit"] = out.loc[long_trigger,  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[short_trigger, "take_profit"] = out.loc[short_trigger, "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(long_trigger,  long_conf,
                                  np.where(short_trigger, short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)

    @staticmethod
    def _rolling_any(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=1).max().astype(bool)
