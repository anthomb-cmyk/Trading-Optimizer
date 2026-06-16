"""
Strategy 2 — Order Block Reversal

An Order Block is the last opposing candle before a significant impulse move.
When price returns to that zone, institutional orders placed there provide
support (bullish OB) or resistance (bearish OB).

This strategy focuses purely on OB mitigation — the moment price touches
back into an OB zone after a confirmed displacement. Unlike FVG Retest (S3),
no imbalance gap is required; the OB zone itself is the entry trigger.

Logic:
  1. Find displacement candles (strong body + large range).
  2. Walk back to identify the last opposing candle = the Order Block.
  3. Track the OB zone (high/low of that candle) until it is "mitigated"
     (price closes through the 50% level of the OB).
  4. On first touch of the OB zone: look for a reversal candle trigger.
  5. Enter with stop beyond the OB extreme.

Confluence checklist:
  [1] Price inside OB zone (low–high of OB candle ± tolerance)   ← required
  [2] OB not yet mitigated (price hasn't closed through OB 50%)
  [3] Reversal candle at OB (bullish pin/engulf for bull OB)
  [4] Daily bias aligned (S10 output feeds here)
  [5] Kill zone timing (S6 output feeds here)
  [6] FVG or BOS on the displacement leg that created the OB
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class OrderBlockReversal(BaseStrategy):
    name         = "order_block_reversal"
    system_role  = "primary"
    required_timeframes = ["15m", "5m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "ob_lookback":           20,
            "ob_body_pct":           0.60,
            "displacement_atr_mult": 0.8,
            "max_ob_age_bars":       30,
            "zone_tolerance_atr":    0.3,    # how close to OB counts as "in zone"
            "mitigation_pct":        0.50,   # OB invalidated if price closes past this level
            "reversal_body_pct":     0.50,   # reversal candle must have this body ratio
            "atr_sl_mult":           0.8,    # SL just beyond OB extreme
            "rr_ratio":              2.5,
            "fvg_min_atr_mult":      0.25,
            "kill_zones":            ["london", "ny_am"],
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "ob_lookback":           trial.suggest_int("ob_lookback",           10, 40),
            "ob_body_pct":           trial.suggest_float("ob_body_pct",         0.45, 0.80),
            "displacement_atr_mult": trial.suggest_float("ob_disp_atr_mult",    0.5, 2.0),
            "max_ob_age_bars":       trial.suggest_int("ob_max_age_bars",       15, 50),
            "zone_tolerance_atr":    trial.suggest_float("ob_zone_tol_atr",     0.1, 0.6),
            "mitigation_pct":        trial.suggest_float("ob_mitigation_pct",   0.30, 0.70),
            "reversal_body_pct":     trial.suggest_float("ob_reversal_body_pct",0.40, 0.75),
            "atr_sl_mult":           trial.suggest_float("ob_atr_sl_mult",      0.4, 1.5),
            "rr_ratio":              trial.suggest_float("ob_rr_ratio",          1.5, 4.0),
            "fvg_min_atr_mult":      trial.suggest_float("ob_fvg_min_atr_mult", 0.15, 0.6),
        }

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr = df["atr"].ffill()
        tol = atr * p["zone_tolerance_atr"]

        # ── Step 1: Detect OBs ────────────────────────────────────────────────
        bull_ob, bear_ob = self.detect_order_blocks(
            df,
            lookback              = p["ob_lookback"],
            body_pct              = p["ob_body_pct"],
            displacement_atr_mult = p["displacement_atr_mult"],
        )

        # OB zone boundaries — forward-filled from the OB candle
        bull_ob_high = df["high"].where(bull_ob).ffill()
        bull_ob_low  = df["low"].where(bull_ob).ffill()
        bear_ob_high = df["high"].where(bear_ob).ffill()
        bear_ob_low  = df["low"].where(bear_ob).ffill()

        # OB midpoint for mitigation check
        bull_ob_mid  = (bull_ob_high + bull_ob_low) / 2
        bear_ob_mid  = (bear_ob_high + bear_ob_low) / 2

        # Age of last OB
        bull_ob_age  = self._bars_since(bull_ob)
        bear_ob_age  = self._bars_since(bear_ob)
        age          = p["max_ob_age_bars"]

        # ── Step 2: Mitigation tracking ───────────────────────────────────────
        # Bullish OB mitigated when price closes below OB midpoint
        bull_mitigated = df["close"] < bull_ob_mid
        bear_mitigated = df["close"] > bear_ob_mid

        # Running "has been mitigated since last OB formed?"
        bull_ob_valid = ~self._rolling_any(bull_mitigated, age) & (bull_ob_age <= age) & (bull_ob_age > 0)
        bear_ob_valid = ~self._rolling_any(bear_mitigated, age) & (bear_ob_age <= age) & (bear_ob_age > 0)

        # ── Step 3: Price in OB zone ──────────────────────────────────────────
        in_bull_zone = (
            bull_ob_valid &
            (df["low"]  <= bull_ob_high + tol) &
            (df["high"] >= bull_ob_low  - tol)
        )
        in_bear_zone = (
            bear_ob_valid &
            (df["high"] >= bear_ob_low  - tol) &
            (df["low"]  <= bear_ob_high + tol)
        )

        # ── Step 4: Reversal candle trigger ───────────────────────────────────
        body_ratio = df["body"] / df["range"].replace(0, np.nan)
        rev_bull   = df["bullish"]  & (body_ratio >= p["reversal_body_pct"])
        rev_bear   = ~df["bullish"] & (body_ratio >= p["reversal_body_pct"])

        # ── Step 5: Confluence ────────────────────────────────────────────────
        bull_fvg, bear_fvg = self.detect_fvg(df, p["fvg_min_atr_mult"])
        fvg_bull_w = self._rolling_any(bull_fvg, age)
        fvg_bear_w = self._rolling_any(bear_fvg, age)

        bull_bos, bear_bos = self.detect_bos(df, p["ob_lookback"])

        in_kz  = self.kill_zone_mask(df, p.get("kill_zones", ["london", "ny_am"]))
        bias   = self.htf_bias(df)

        long_conf = (
            in_bull_zone.astype(int) +           # [1] price in OB zone
            bull_ob_valid.astype(int) +           # [2] OB not mitigated
            rev_bull.astype(int) +                # [3] reversal candle
            (bias == 1).astype(int) +             # [4] daily/HTF bias
            in_kz.astype(int) +                   # [5] kill zone
            (fvg_bull_w | bull_bos).astype(int)   # [6] FVG or BOS on leg
        )
        short_conf = (
            in_bear_zone.astype(int) +
            bear_ob_valid.astype(int) +
            rev_bear.astype(int) +
            (bias == -1).astype(int) +
            in_kz.astype(int) +
            (fvg_bear_w | bear_bos).astype(int)
        )

        long_trigger  = in_bull_zone & rev_bull & (long_conf  >= self.min_confluence)
        short_trigger = in_bear_zone & rev_bear & (short_conf >= self.min_confluence)

        out["long_signal"]  = long_trigger
        out["short_signal"] = short_trigger

        out.loc[long_trigger,  "entry_price"] = df.loc[long_trigger,  "close"]
        out.loc[short_trigger, "entry_price"] = df.loc[short_trigger, "close"]

        atr_sl = atr * p["atr_sl_mult"]
        # SL just beyond the OB extreme
        out.loc[long_trigger,  "stop_loss"] = bull_ob_low.loc[long_trigger]  - atr_sl.loc[long_trigger]
        out.loc[short_trigger, "stop_loss"] = bear_ob_high.loc[short_trigger] + atr_sl.loc[short_trigger]

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[long_trigger,  "take_profit"] = out.loc[long_trigger,  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[short_trigger, "take_profit"] = out.loc[short_trigger, "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(long_trigger,  long_conf,
                                  np.where(short_trigger, short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)
