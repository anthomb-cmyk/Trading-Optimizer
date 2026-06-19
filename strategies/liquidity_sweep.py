"""
Strategy 1 — Liquidity Sweep Reversal

Logic:
  1. Identify equal highs (buy-side liquidity) and equal lows (sell-side liquidity)
     using a rolling window to find clustered swing highs/lows within a % tolerance.
  2. When price wicks above equal highs and then closes back below → bearish setup.
  3. When price wicks below equal lows and then closes back above → bullish setup.
  4. Score confluence: HTF bias, FVG after sweep, kill zone, volume spike, RSI extreme.

Confluence checklist (each = +1):
  [1] Price swept liquidity level (wick beyond, close inside)  ← required
  [2] HTF EMA-200 bias aligns with trade direction
  [3] FVG formed in same direction on bar after sweep
  [4] Inside kill zone
  [5] Volume spike on sweep candle (>= vol_spike_mult * avg)
  [6] RSI at extreme (>70 for short sweep, <30 for long sweep)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.settings import RISK, SMC
from strategies.base_strategy import BaseStrategy


class LiquiditySweepReversal(BaseStrategy):
    name        = "liquidity_sweep_reversal"
    system_role = "primary"
    required_timeframes = ["15m", "5m"]

    # ── Default params ────────────────────────────────────────────────────────

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "sweep_lookback":    30,
            "equal_pct":         0.0015,
            "vol_spike_mult":    1.5,
            "atr_sl_mult":       1.2,
            "rr_ratio":          2.0,
            "rsi_extreme_long":  35,
            "rsi_extreme_short": 65,
            "kill_zones":        ["london", "ny_am"],
            "fvg_min_atr_mult":  0.3,
        }

    # ── Optuna param space ────────────────────────────────────────────────────

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "sweep_lookback":    trial.suggest_int("ls_sweep_lookback",    15, 60),
            "equal_pct":         trial.suggest_float("ls_equal_pct",       0.0005, 0.004, log=True),
            "vol_spike_mult":    trial.suggest_float("ls_vol_spike_mult",   1.0, 3.0),
            "atr_sl_mult":       trial.suggest_float("ls_atr_sl_mult",      0.8, 2.5),
            "rr_ratio":          trial.suggest_float("ls_rr_ratio",         1.5, 4.0),
            "rsi_extreme_long":  trial.suggest_int("ls_rsi_extreme_long",   25, 45),
            "rsi_extreme_short": trial.suggest_int("ls_rsi_extreme_short",  55, 75),
            "fvg_min_atr_mult":  trial.suggest_float("ls_fvg_min_atr_mult", 0.2, 0.8),
        }

    # ── Main signal generation ────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        lb       = p["sweep_lookback"]
        eq_pct   = p["equal_pct"]
        atr      = df["atr"].ffill()

        # ── Step 1: Find equal highs and equal lows ──────────────────────────
        eq_highs = self._equal_levels(df["swing_high_price"], df["swing_high"], lb, eq_pct)
        eq_lows  = self._equal_levels(df["swing_low_price"],  df["swing_low"],  lb, eq_pct)

        # ── Step 2: Detect sweeps ────────────────────────────────────────────
        # Bullish sweep: wick below equal lows, close back above
        bull_sweep = (
            eq_lows.shift(1).fillna(False) &
            (df["low"] < df["low"].shift(1)) &
            (df["close"] > df["low"].shift(1))
        )
        # Bearish sweep: wick above equal highs, close back below
        bear_sweep = (
            eq_highs.shift(1).fillna(False) &
            (df["high"] > df["high"].shift(1)) &
            (df["close"] < df["high"].shift(1))
        )

        # ── Step 3: Confluence scoring ────────────────────────────────────────
        bull_fvg, bear_fvg = self.detect_fvg(df, p["fvg_min_atr_mult"])
        bias   = self.htf_bias(df)
        in_kz  = self.kill_zone_mask(df, p.get("kill_zones", ["london", "ny_am"]))
        vol_ok = self.volume_filter(df, p["vol_spike_mult"])
        vol_ok_loose = self.volume_filter(df, 1.0)

        # RSI extremes
        rsi = df.get("rsi", pd.Series(50, index=df.index)).fillna(50)
        rsi_long  = rsi <= p["rsi_extreme_long"]
        rsi_short = rsi >= p["rsi_extreme_short"]

        # Build confluence counts
        long_conf = (
            bull_sweep.astype(int) +            # [1] sweep (required)
            (bias == 1).astype(int) +           # [2] HTF bullish
            bull_fvg.astype(int) +              # [3] FVG up
            in_kz.astype(int) +                 # [4] kill zone
            vol_ok.astype(int) +                # [5] volume spike
            rsi_long.astype(int)                # [6] RSI oversold
        )
        short_conf = (
            bear_sweep.astype(int) +
            (bias == -1).astype(int) +
            bear_fvg.astype(int) +
            in_kz.astype(int) +
            vol_ok.astype(int) +
            rsi_short.astype(int)
        )

        # ── Step 4: Entry, SL, TP ────────────────────────────────────────────
        out["long_signal"]  = bull_sweep & (long_conf  >= self.min_confluence)
        out["short_signal"] = bear_sweep & (short_conf >= self.min_confluence)

        out.loc[out["long_signal"],  "entry_price"] = df.loc[out["long_signal"],  "close"]
        out.loc[out["short_signal"], "entry_price"] = df.loc[out["short_signal"], "close"]

        atr_sl = atr * p["atr_sl_mult"]
        out.loc[out["long_signal"],  "stop_loss"] = out.loc[out["long_signal"],  "entry_price"] - atr_sl
        out.loc[out["short_signal"], "stop_loss"] = out.loc[out["short_signal"], "entry_price"] + atr_sl

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[out["long_signal"],  "take_profit"] = out.loc[out["long_signal"],  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[out["short_signal"], "take_profit"] = out.loc[out["short_signal"], "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(out["long_signal"],  long_conf,
                                  np.where(out["short_signal"], short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _equal_levels(
        price: pd.Series,
        swing_mask: pd.Series,
        lookback: int,
        tolerance_pct: float,
    ) -> pd.Series:
        """
        Return True where *price* at a swing point is within *tolerance_pct*
        of another swing point in the previous *lookback* bars.
        Indicates a liquidity pool (double top / double bottom area).
        """
        values = price.values
        swings = swing_mask.values
        n      = len(values)
        result = np.zeros(n, dtype=bool)

        for i in range(lookback, n):
            if not swings[i]:
                continue
            ref = values[i]
            if ref == 0:
                continue
            window_swings = swings[max(0, i - lookback): i]
            window_prices = values[max(0, i - lookback): i]
            comparisons   = window_prices[window_swings]
            if len(comparisons) == 0:
                continue
            pct_diff = np.abs(comparisons - ref) / ref
            if np.any(pct_diff <= tolerance_pct):
                result[i] = True

        return pd.Series(result, index=price.index)
