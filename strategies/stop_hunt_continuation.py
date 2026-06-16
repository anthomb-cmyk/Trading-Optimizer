"""
Strategy 10 — Stop Hunt Continuation

Logic:
  Opposite of the Liquidity Sweep Reversal. Here, instead of fading the
  stop hunt, we trade WITH the breakout direction after a brief stop-hunt
  that flushes weak hands — then continue in the original direction.

  The "stop hunt continuation" pattern:
    1. Price is trending (HH/HL or LL/LH structure established).
    2. A key level (previous swing, day's high/low, round number) is targeted.
    3. Price briefly breaks BELOW (for bull) or ABOVE (for bear) that level,
       triggering stop losses — the "hunt."
    4. Instead of reversing, price quickly reclaims the level and continues
       in the trend direction.
    5. This is NOT a reversal — it's a continuation after weak-hand clearance.

  Key differentiator from Liquidity Sweep Reversal:
    - SHC requires an established trend BEFORE the sweep.
    - SHC enters IN the trend direction.
    - LSR fades the sweep (counter-trend).

Confluence checklist:
  [1] Established trend (swing structure) in entry direction      ← required
  [2] Level sweep (wick beyond key level then reclaim)
  [3] Quick reclaim (close back above/below level within x bars)  ← required
  [4] Momentum candle after reclaim (body > body_pct * range)
  [5] HTF bias aligned
  [6] Volume spike on reclaim candle or kill zone timing
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class StopHuntContinuation(BaseStrategy):
    name        = "stop_hunt_continuation"
    system_role = "confirmation"
    required_timeframes = ["15m", "5m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "trend_lookback":       40,
            "level_lookback":       30,     # bars to define key S/R level
            "sweep_wick_atr_mult":  0.3,    # min wick beyond level
            "reclaim_bars":         3,      # bars to reclaim level after sweep
            "momentum_body_pct":    0.55,   # momentum candle body ratio
            "momentum_atr_mult":    0.5,    # momentum candle must be >= x * ATR
            "atr_sl_mult":          1.0,
            "rr_ratio":             2.0,
            "kill_zones":           ["london", "ny_am"],
            "fvg_min_atr_mult":     0.25,
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "trend_lookback":      trial.suggest_int("shc_trend_lookback",     20, 80),
            "level_lookback":      trial.suggest_int("shc_level_lookback",     15, 60),
            "sweep_wick_atr_mult": trial.suggest_float("shc_sweep_wick",       0.1, 0.8),
            "reclaim_bars":        trial.suggest_int("shc_reclaim_bars",       1, 6),
            "momentum_body_pct":   trial.suggest_float("shc_momentum_body",    0.45, 0.80),
            "momentum_atr_mult":   trial.suggest_float("shc_momentum_atr",     0.3, 1.2),
            "atr_sl_mult":         trial.suggest_float("shc_atr_sl_mult",      0.6, 2.0),
            "rr_ratio":            trial.suggest_float("shc_rr_ratio",         1.5, 3.5),
            "fvg_min_atr_mult":    trial.suggest_float("shc_fvg_min_atr",      0.15, 0.6),
        }

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr = df["atr"].ffill()
        lb  = p["trend_lookback"]

        # ── Step 1: Trend identification via swing structure ──────────────────
        swing_high_price = df["high"].where(df["swing_high"]).ffill()
        swing_low_price  = df["low"].where(df["swing_low"]).ffill()

        # Bullish trend: current swing high > previous swing high AND
        #                current swing low  > previous swing low
        prev_sh = df["high"].where(df["swing_high"]).shift(1).ffill()
        prev_sl = df["low"].where(df["swing_low"]).shift(1).ffill()

        bull_trend = (swing_high_price > prev_sh) & (swing_low_price > prev_sl)
        bear_trend = (swing_high_price < prev_sh) & (swing_low_price < prev_sl)

        # Rolling trend confirmation over lb bars
        bull_trend_confirmed = bull_trend.rolling(lb).sum() > (lb * 0.6)
        bear_trend_confirmed = bear_trend.rolling(lb).sum() > (lb * 0.6)

        # ── Step 2: Key level identification ─────────────────────────────────
        # Use recent swing lows as support (for bull continuation)
        # and recent swing highs as resistance (for bear continuation)
        key_support    = df["low"].rolling(p["level_lookback"]).min().shift(1)
        key_resistance = df["high"].rolling(p["level_lookback"]).max().shift(1)

        min_wick = atr * p["sweep_wick_atr_mult"]

        # ── Step 3: Sweep + reclaim ───────────────────────────────────────────
        # Bull continuation: wick below support, close back above
        bull_sweep = (
            bull_trend_confirmed &
            (df["low"]   < key_support  - min_wick) &
            (df["close"] > key_support)
        )
        # Bear continuation: wick above resistance, close back below
        bear_sweep = (
            bear_trend_confirmed &
            (df["high"]  > key_resistance + min_wick) &
            (df["close"] < key_resistance)
        )

        # ── Step 4: Momentum candle after reclaim ─────────────────────────────
        body_pct  = df["body"] / df["range"].replace(0, np.nan)
        momentum  = (body_pct >= p["momentum_body_pct"]) & (df["range"] >= atr * p["momentum_atr_mult"])
        mom_bull  = momentum & df["bullish"]
        mom_bear  = momentum & ~df["bullish"]

        # Momentum must occur within reclaim_bars of the sweep
        sweep_bull_recent = self._rolling_any(bull_sweep, p["reclaim_bars"] + 1)
        sweep_bear_recent = self._rolling_any(bear_sweep, p["reclaim_bars"] + 1)

        long_trigger  = mom_bull & sweep_bull_recent & ~bull_sweep
        short_trigger = mom_bear & sweep_bear_recent & ~bear_sweep

        # ── Step 5: Confluence ────────────────────────────────────────────────
        bull_fvg, bear_fvg = self.detect_fvg(df, p["fvg_min_atr_mult"])
        fvg_bull_w = self._rolling_any(bull_fvg, 5)
        fvg_bear_w = self._rolling_any(bear_fvg, 5)

        bias  = self.htf_bias(df)
        in_kz = self.kill_zone_mask(df, p.get("kill_zones", ["london", "ny_am"]))
        vol   = self.volume_filter(df)

        long_conf = (
            bull_trend_confirmed.astype(int) +      # [1] established bull trend
            sweep_bull_recent.astype(int) +          # [2] level swept
            long_trigger.astype(int) +              # [3] reclaimed + momentum
            fvg_bull_w.astype(int) +                # [4] FVG
            (bias == 1).astype(int) +               # [5] HTF bullish
            (in_kz | vol).astype(int)               # [6] kill zone or volume
        )
        short_conf = (
            bear_trend_confirmed.astype(int) +
            sweep_bear_recent.astype(int) +
            short_trigger.astype(int) +
            fvg_bear_w.astype(int) +
            (bias == -1).astype(int) +
            (in_kz | vol).astype(int)
        )

        out["long_signal"]  = long_trigger  & (long_conf  >= self.min_confluence)
        out["short_signal"] = short_trigger & (short_conf >= self.min_confluence)

        out.loc[out["long_signal"],  "entry_price"] = df.loc[out["long_signal"],  "close"]
        out.loc[out["short_signal"], "entry_price"] = df.loc[out["short_signal"], "close"]

        atr_sl = atr * p["atr_sl_mult"]
        out.loc[out["long_signal"],  "stop_loss"] = (
            df.loc[out["long_signal"], "low"] - atr_sl.loc[out["long_signal"]]
        )
        out.loc[out["short_signal"], "stop_loss"] = (
            df.loc[out["short_signal"], "high"] + atr_sl.loc[out["short_signal"]]
        )

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[out["long_signal"],  "take_profit"] = out.loc[out["long_signal"],  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[out["short_signal"], "take_profit"] = out.loc[out["short_signal"], "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(out["long_signal"],  long_conf,
                                  np.where(out["short_signal"], short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)

    @staticmethod
    def _rolling_any(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=1).max().astype(bool)
