"""
Strategy 6 — OTE Fibonacci Entry

Logic:
  ICT's Optimal Trade Entry (OTE) uses the 0.618–0.786 Fibonacci retracement
  as the highest-probability pullback zone after a confirmed swing.

  Steps:
    1. Identify a clean, significant swing (must be confirmed BOS).
    2. After the swing, wait for price to retrace into the OTE band.
    3. Look for a reversal trigger at OTE: bullish/bearish candle pattern,
       Order Block, or FVG within the OTE band.
    4. Entry at OTE trigger, stop below/above the swing origin.

  Bullish OTE: strong swing UP (swing low → swing high), retrace to 0.618–0.786,
               enter long with stop below the swing low.
  Bearish OTE: strong swing DOWN, retrace to 0.618–0.786 from the top,
               enter short with stop above the swing high.

Confluence checklist:
  [1] Price inside OTE band (0.618 – 0.786 of the qualifying swing)  ← required
  [2] Retracement >= minimum (price actually pulled back enough)
  [3] Reversal candle or Order Block at OTE level
  [4] HTF bias aligned with swing direction
  [5] Inside kill zone
  [6] FVG present within OTE band
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class OTEFibonacciEntry(BaseStrategy):
    name        = "ote_fibonacci_entry"
    system_role = "confirmation"
    required_timeframes = ["15m", "5m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "swing_lookback":      15,
            "swing_min_atr_mult":  2.0,    # swing must span >= x * ATR
            "ote_low_fib":         0.618,
            "ote_high_fib":        0.786,
            "retracement_min":     0.382,  # minimum retracement to qualify
            "ote_tolerance_atr":   0.4,
            "reversal_body_pct":   0.5,    # reversal candle body ratio
            "ob_lookback":         15,
            "ob_body_pct":         0.55,
            "atr_sl_mult":         0.5,    # tight — stop at swing origin
            "rr_ratio":            3.0,
            "kill_zones":          ["london", "ny_am"],
            "fvg_min_atr_mult":    0.25,
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "swing_lookback":     trial.suggest_int("ote_swing_lookback",     8, 30),
            "swing_min_atr_mult": trial.suggest_float("ote_swing_min_atr",    1.0, 4.0),
            "ote_low_fib":        trial.suggest_float("ote_low_fib",          0.55, 0.68),
            "ote_high_fib":       trial.suggest_float("ote_high_fib",         0.72, 0.85),
            "retracement_min":    trial.suggest_float("ote_retracement_min",  0.25, 0.55),
            "ote_tolerance_atr":  trial.suggest_float("ote_tolerance_atr",    0.2, 0.8),
            "reversal_body_pct":  trial.suggest_float("ote_reversal_body_pct",0.40, 0.75),
            "ob_lookback":        trial.suggest_int("ote_ob_lookback",        8, 30),
            "ob_body_pct":        trial.suggest_float("ote_ob_body_pct",      0.45, 0.80),
            "atr_sl_mult":        trial.suggest_float("ote_atr_sl_mult",      0.3, 1.5),
            "rr_ratio":           trial.suggest_float("ote_rr_ratio",         2.0, 5.0),
            "fvg_min_atr_mult":   trial.suggest_float("ote_fvg_min_atr_mult", 0.15, 0.6),
        }

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr = df["atr"].ffill()
        lb  = p["swing_lookback"]
        tol = atr * p["ote_tolerance_atr"]

        # ── Step 1: Identify qualifying swings ────────────────────────────────
        # A qualifying bullish swing: swing low followed by swing high with
        # sufficient range AND a BOS (close above swing high confirms the move).
        swing_high_price = df["high"].where(df["swing_high"]).ffill()
        swing_low_price  = df["low"].where(df["swing_low"]).ffill()

        # Minimum swing size
        swing_up_size   = swing_high_price - swing_low_price.shift(lb)
        swing_down_size = swing_low_price  - swing_high_price.shift(lb)

        min_swing = atr * p["swing_min_atr_mult"]
        valid_bull_swing = swing_up_size   >= min_swing
        valid_bear_swing = swing_down_size.abs() >= min_swing

        # BOS to confirm swing direction
        bull_bos, bear_bos = self.detect_bos(df, lb, confirm_bars=1)

        # ── Step 2: OTE band calculation ─────────────────────────────────────
        # Bullish OTE: retracement of a bullish swing
        #   origin = swing_low, peak = swing_high
        #   OTE band: [peak - ote_high * size, peak - ote_low * size]
        bull_origin = swing_low_price
        bull_peak   = swing_high_price
        bull_size   = (bull_peak - bull_origin).clip(lower=0)

        ote_bull_lo = bull_peak - p["ote_high_fib"] * bull_size
        ote_bull_hi = bull_peak - p["ote_low_fib"]  * bull_size

        bear_origin = swing_high_price
        bear_peak   = swing_low_price
        bear_size   = (bear_origin - bear_peak).clip(lower=0)

        ote_bear_lo = bear_peak  + p["ote_low_fib"]  * bear_size
        ote_bear_hi = bear_peak  + p["ote_high_fib"] * bear_size

        # Price in OTE band
        in_ote_bull = (
            (df["low"]  <= ote_bull_hi + tol) &
            (df["high"] >= ote_bull_lo - tol) &
            valid_bull_swing & bull_bos
        )
        in_ote_bear = (
            (df["high"] >= ote_bear_lo - tol) &
            (df["low"]  <= ote_bear_hi + tol) &
            valid_bear_swing & bear_bos
        )

        # ── Step 3: Reversal trigger at OTE ──────────────────────────────────
        body_pct = df["body"] / df["range"].replace(0, np.nan)
        rev_bull = df["bullish"] & (body_pct >= p["reversal_body_pct"])   # bullish engulf/hammer
        rev_bear = ~df["bullish"] & (body_pct >= p["reversal_body_pct"])  # bearish

        bull_ob, bear_ob = self.detect_order_blocks(df, p["ob_lookback"], p["ob_body_pct"])
        bull_ob_w = self._rolling_any(bull_ob, 8)
        bear_ob_w = self._rolling_any(bear_ob, 8)

        # ── Step 4: Confluence ────────────────────────────────────────────────
        bull_fvg, bear_fvg = self.detect_fvg(df, p["fvg_min_atr_mult"])
        fvg_bull_w = self._rolling_any(bull_fvg, 8)
        fvg_bear_w = self._rolling_any(bear_fvg, 8)

        bias  = self.htf_bias(df)
        in_kz = self.kill_zone_mask(df, p.get("kill_zones", ["london", "ny_am"]))

        long_conf = (
            in_ote_bull.astype(int) +            # [1] in OTE band
            valid_bull_swing.astype(int) +        # [2] swing qualifies
            (rev_bull | bull_ob_w).astype(int) +  # [3] reversal trigger or OB
            (bias == 1).astype(int) +             # [4] HTF bullish
            in_kz.astype(int) +                   # [5] kill zone
            fvg_bull_w.astype(int)                # [6] FVG in band
        )
        short_conf = (
            in_ote_bear.astype(int) +
            valid_bear_swing.astype(int) +
            (rev_bear | bear_ob_w).astype(int) +
            (bias == -1).astype(int) +
            in_kz.astype(int) +
            fvg_bear_w.astype(int)
        )

        long_trigger  = in_ote_bull & rev_bull & (long_conf  >= self.min_confluence)
        short_trigger = in_ote_bear & rev_bear & (short_conf >= self.min_confluence)

        out["long_signal"]  = long_trigger
        out["short_signal"] = short_trigger

        out.loc[long_trigger,  "entry_price"] = df.loc[long_trigger,  "close"]
        out.loc[short_trigger, "entry_price"] = df.loc[short_trigger, "close"]

        # SL at swing origin with small ATR buffer
        atr_sl = atr * p["atr_sl_mult"]
        out.loc[long_trigger,  "stop_loss"] = bull_origin.loc[long_trigger]  - atr_sl.loc[long_trigger]
        out.loc[short_trigger, "stop_loss"] = bear_origin.loc[short_trigger] + atr_sl.loc[short_trigger]

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[long_trigger,  "take_profit"] = out.loc[long_trigger,  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[short_trigger, "take_profit"] = out.loc[short_trigger, "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(long_trigger,  long_conf,
                                  np.where(short_trigger, short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)

    @staticmethod
    def _rolling_any(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=1).max().astype(bool)
