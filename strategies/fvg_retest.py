"""
Strategy 3 — FVG Retest

A Fair Value Gap (FVG) is a 3-candle imbalance where price moved so fast that
no two-sided trading occurred — leaving an "unfilled" zone that the market
frequently returns to before resuming the original direction.

This strategy tracks open FVG zones and enters when price retests them,
distinguishing between partial and full fills and preferring entry at the
FVG edge (50% fill) rather than waiting for a complete fill.

FVG Zone definitions:
  Bullish FVG: bottom = candle[i-2].high, top = candle[i].low
               (price gapped up — unfilled region below current price)
  Bearish FVG: bottom = candle[i].high,   top = candle[i-2].low
               (price gapped down — unfilled region above current price)

FVG lifecycle:
  OPEN     → price has not yet touched the FVG zone
  RETEST   → price is touching or inside the FVG zone (entry zone)
  FILLED   → price closed through the FVG bottom (bull) or top (bear) — invalidated

Logic:
  1. Detect FVG formation (3-candle pattern).
  2. Track zone boundaries and lifecycle state.
  3. When price retests a OPEN FVG from the correct side → entry trigger.
  4. Entry at FVG edge or midpoint; stop beyond FVG opposite edge.

Confluence checklist:
  [1] Open (unfilled) FVG zone present                            ← required
  [2] Price entering FVG zone from correct side (first touch)     ← required
  [3] FVG aligned with daily bias direction (S10 output)
  [4] Kill zone timing (S6 output)
  [5] Order Block within or adjacent to FVG zone
  [6] Volume contraction during retest (low-vol pullback = healthy)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class FVGRetest(BaseStrategy):
    name        = "fvg_retest"
    system_role = "primary"
    required_timeframes = ["15m", "5m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "fvg_min_atr_mult":    0.25,
            "max_fvg_age_bars":    40,
            "entry_at":            "edge",    # "edge" | "mid" | "far_edge"
            "zone_tol_atr":        0.2,
            "ob_lookback":         15,
            "ob_body_pct":         0.55,
            "vol_contraction_mult":0.8,       # retest volume < x * avg = healthy pullback
            "atr_sl_mult":         0.5,
            "rr_ratio":            3.0,
            "kill_zones":          ["london", "ny_am"],
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "fvg_min_atr_mult":    trial.suggest_float("fvg_min_atr_mult",     0.15, 0.60),
            "max_fvg_age_bars":    trial.suggest_int("fvg_max_age_bars",       15, 60),
            "entry_at":            trial.suggest_categorical("fvg_entry_at", ["edge", "mid", "far_edge"]),
            "zone_tol_atr":        trial.suggest_float("fvg_zone_tol_atr",     0.1, 0.5),
            "ob_lookback":         trial.suggest_int("fvg_ob_lookback",        8, 30),
            "ob_body_pct":         trial.suggest_float("fvg_ob_body_pct",      0.45, 0.80),
            "vol_contraction_mult":trial.suggest_float("fvg_vol_contraction",  0.5, 1.2),
            "atr_sl_mult":         trial.suggest_float("fvg_atr_sl_mult",      0.3, 1.2),
            "rr_ratio":            trial.suggest_float("fvg_rr_ratio",          2.0, 5.0),
        }

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr = df["atr"].ffill()
        tol = atr * p["zone_tol_atr"]
        age = p["max_fvg_age_bars"]

        # ── Step 1: Detect FVG zones ──────────────────────────────────────────
        bull_top, bull_bot, bear_top, bear_bot = self._fvg_zones(df, p["fvg_min_atr_mult"])

        # ── Step 2: FVG fill tracking ─────────────────────────────────────────
        # Bullish FVG is "filled" (invalidated) when price closes below its bottom
        bull_filled = df["close"] < bull_bot
        bear_filled = df["close"] > bear_top

        # Last open (unfilled) FVG zone — forward-fill until filled
        # Approach: track the last FVG that hasn't been filled yet
        bull_open_top, bull_open_bot = self._track_open_fvg(
            bull_top, bull_bot, bull_filled, age
        )
        bear_open_top, bear_open_bot = self._track_open_fvg(
            bear_top, bear_bot, bear_filled, age
        )

        bull_open_mid = (bull_open_top + bull_open_bot) / 2
        bear_open_mid = (bear_open_top + bear_open_bot) / 2

        # ── Step 3: Price retesting FVG zone ─────────────────────────────────
        # Bullish FVG retest: price pulls back DOWN into the FVG zone from above
        in_bull_fvg = (
            bull_open_top.notna() &
            (df["low"]  <= bull_open_top + tol) &
            (df["high"] >= bull_open_bot - tol)
        )
        in_bear_fvg = (
            bear_open_top.notna() &
            (df["high"] >= bear_open_bot - tol) &
            (df["low"]  <= bear_open_top + tol)
        )

        # Detect the genuine first-touch per zone using _track_first_touch.
        # This tracks whether the current zone has been entered before; the
        # first bar where price enters is the only one flagged True.
        bull_first_touch_raw = self._track_first_touch(in_bull_fvg, bull_top)
        bear_first_touch_raw = self._track_first_touch(in_bear_fvg, bear_top)

        # Entry price depends on mode
        if p["entry_at"] == "edge":
            long_entry_ref  = bull_open_top   # enter at top of bull FVG (50% fill)
            short_entry_ref = bear_open_bot   # enter at bottom of bear FVG
        elif p["entry_at"] == "mid":
            long_entry_ref  = bull_open_mid
            short_entry_ref = bear_open_mid
        else:  # far_edge — deepest fill, most patient
            long_entry_ref  = bull_open_bot
            short_entry_ref = bear_open_top

        # ── Step 4: Confluence ────────────────────────────────────────────────
        bull_ob, bear_ob, *_ = self.detect_order_blocks(df, p["ob_lookback"], p["ob_body_pct"])
        ob_bull_w = self._rolling_any(bull_ob, age)
        ob_bear_w = self._rolling_any(bear_ob, age)

        in_kz  = self.kill_zone_mask(df, p.get("kill_zones", ["london", "ny_am"]))
        bias   = self.htf_bias(df)

        # Volume contraction during retest = healthy pullback
        vol_low = self.volume_filter(df, 0.01)  # just not zero
        vol_contract = (
            ~self.volume_filter(df, p["vol_contraction_mult"])
            if "vol_ma" in df.columns and not df["vol_ma"].isna().all()
            else pd.Series(True, index=df.index)
        )

        bull_first_touch = bull_first_touch_raw
        bear_first_touch = bear_first_touch_raw

        # ── Step 4 continued: Confluence ──────────────────────────────────────
        long_conf = (
            in_bull_fvg.astype(int) +            # [1] open FVG exists
            bull_first_touch.astype(int) +        # [2] first touch of FVG (entering from correct side)
            (bias == 1).astype(int) +            # [3] daily bias bullish
            in_kz.astype(int) +                  # [4] kill zone
            ob_bull_w.astype(int) +              # [5] OB at FVG
            vol_contract.astype(int)             # [6] volume contraction
        )
        short_conf = (
            in_bear_fvg.astype(int) +
            bear_first_touch.astype(int) +        # [2] first touch of FVG (entering from correct side)
            (bias == -1).astype(int) +
            in_kz.astype(int) +
            ob_bear_w.astype(int) +
            vol_contract.astype(int)
        )

        # Entry: first touch of the FVG zone, inside a kill zone, HTF bias aligned,
        # with confluence >= threshold.
        # Kill zone + bias alignment as hard gates mirrors the docstring intent:
        # [3] FVG aligned with daily bias and [4] kill zone timing are key required filters.
        long_trigger  = bull_first_touch & in_kz & (bias == 1) & (long_conf  >= self.min_confluence)
        short_trigger = bear_first_touch & in_kz & (bias == -1) & (short_conf >= self.min_confluence)

        out["long_signal"]  = long_trigger
        out["short_signal"] = short_trigger

        # Entry at reference level (clipped to actual close for market orders)
        out.loc[long_trigger,  "entry_price"] = np.minimum(
            df.loc[long_trigger, "close"],
            long_entry_ref.loc[long_trigger].fillna(df.loc[long_trigger, "close"])
        )
        out.loc[short_trigger, "entry_price"] = np.maximum(
            df.loc[short_trigger, "close"],
            short_entry_ref.loc[short_trigger].fillna(df.loc[short_trigger, "close"])
        )

        atr_sl = atr * p["atr_sl_mult"]
        out.loc[long_trigger,  "stop_loss"] = bull_open_bot.loc[long_trigger]  - atr_sl.loc[long_trigger]
        out.loc[short_trigger, "stop_loss"] = bear_open_top.loc[short_trigger] + atr_sl.loc[short_trigger]

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[long_trigger,  "take_profit"] = out.loc[long_trigger,  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[short_trigger, "take_profit"] = out.loc[short_trigger, "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(long_trigger,  long_conf,
                                  np.where(short_trigger, short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _track_first_touch(
        in_zone: pd.Series,
        zone_new: pd.Series,
    ) -> pd.Series:
        """
        Return a boolean Series that is True ONLY on the first bar price enters
        a given FVG zone (ignoring subsequent bars in the same zone).

        `in_zone`  — True when price is touching/inside the tracked zone.
        `zone_new` — notna() marks bars where a new zone was created (bull_top
                     or bear_top from _fvg_zones); used to reset the touched flag.

        Algorithm: iterate bar-by-bar, tracking per-zone whether it has been
        entered before. A new zone (zone_new.notna()) resets the touched flag.
        """
        in_z      = in_zone.values.astype(bool)
        is_new    = zone_new.notna().values
        n         = len(in_z)
        result    = np.zeros(n, dtype=bool)
        touched   = True   # sentinel: no zone yet, so mark as "touched" to suppress

        for i in range(n):
            if is_new[i]:
                touched = False  # new zone formed — reset; next entry is first touch
            if in_z[i] and not touched:
                result[i] = True
                touched = True   # only the very first entry bar is flagged
        return pd.Series(result, index=in_zone.index)

    @staticmethod
    def _fvg_zones(
        df: pd.DataFrame,
        min_atr_mult: float,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Returns (bull_top, bull_bot, bear_top, bear_bot) — zone boundary Series.
        NaN where no FVG exists on that bar.
        """
        atr     = df["atr"].ffill()
        min_gap = atr * min_atr_mult
        high_2  = df["high"].shift(2)
        low_2   = df["low"].shift(2)

        bull_mask = (high_2 < df["low"])  & ((df["low"]  - high_2) >= min_gap)
        bear_mask = (low_2  > df["high"]) & ((low_2 - df["high"])  >= min_gap)

        bull_top = df["low"].where(bull_mask)    # top of bullish FVG
        bull_bot = high_2.where(bull_mask)        # bottom of bullish FVG
        bear_top = low_2.where(bear_mask)         # top of bearish FVG
        bear_bot = df["high"].where(bear_mask)    # bottom of bearish FVG

        return bull_top, bull_bot, bear_top, bear_bot

    @staticmethod
    def _track_open_fvg(
        top: pd.Series,
        bot: pd.Series,
        filled_mask: pd.Series,
        max_age: int,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Forward-fill the most recent FVG zone that has not been filled.
        Resets to NaN when the fill condition is met or age > max_age.
        """
        top_vals    = top.values.copy().astype(float)
        bot_vals    = bot.values.copy().astype(float)
        filled      = filled_mask.values.astype(bool)
        n           = len(top_vals)

        open_top = np.full(n, np.nan)
        open_bot = np.full(n, np.nan)
        cur_top  = np.nan
        cur_bot  = np.nan
        cur_age  = 9999

        for i in range(n):
            # New FVG formed on this bar
            if not np.isnan(top_vals[i]):
                cur_top = top_vals[i]
                cur_bot = bot_vals[i]
                cur_age = 0
            else:
                cur_age += 1

            # Invalidate if filled or too old
            if filled[i] or cur_age > max_age:
                cur_top = np.nan
                cur_bot = np.nan

            open_top[i] = cur_top
            open_bot[i] = cur_bot

        return (
            pd.Series(open_top, index=top.index),
            pd.Series(open_bot, index=bot.index),
        )
