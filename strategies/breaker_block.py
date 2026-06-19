"""
Strategy 3 — Breaker Block Reversal

Logic:
  1. Find an Order Block (last directional candle before a swing).
  2. Price breaks THROUGH the OB decisively (the OB "fails" and becomes a Breaker).
  3. Price later returns to test the Breaker Block from the opposite side.
  4. The Breaker Block now acts as resistance (bearish) or support (bullish).

  Bullish Breaker: A bearish OB is broken to the downside, then price rallies
  back up and retests it from below as support.
  Bearish Breaker: A bullish OB is broken to the upside, then price falls back
  and retests it from above as resistance.

Confluence checklist:
  [1] Price at Breaker Block retest zone                       ← required
  [2] Strong displacement candle created the break (ATR-qualified)
  [3] HTF bias aligns with Breaker direction
  [4] FVG present in the displacement leg
  [5] Inside kill zone
  [6] Volume confirmation on displacement candle
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class BreakerBlockReversal(BaseStrategy):
    name        = "breaker_block_reversal"
    system_role = "confirmation"
    required_timeframes = ["15m", "1h"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "ob_lookback":          20,
            "ob_body_pct":          0.6,
            "break_atr_mult":       1.5,    # displacement must be >= x * ATR
            "max_breaker_age_bars": 40,
            "retest_tolerance_atr": 0.3,    # how close to breaker = "retest"
            "atr_sl_mult":          1.0,
            "rr_ratio":             2.5,
            "kill_zones":           ["london", "ny_am"],
            "fvg_min_atr_mult":     0.3,
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "ob_lookback":          trial.suggest_int("bb_ob_lookback",          10, 40),
            "ob_body_pct":          trial.suggest_float("bb_ob_body_pct",        0.45, 0.80),
            "break_atr_mult":       trial.suggest_float("bb_break_atr_mult",     0.8, 2.5),
            "max_breaker_age_bars": trial.suggest_int("bb_max_breaker_age_bars", 20, 60),
            "retest_tolerance_atr": trial.suggest_float("bb_retest_tolerance",   0.1, 0.6),
            "atr_sl_mult":          trial.suggest_float("bb_atr_sl_mult",        0.6, 2.0),
            "rr_ratio":             trial.suggest_float("bb_rr_ratio",           1.5, 4.0),
            "fvg_min_atr_mult":     trial.suggest_float("bb_fvg_min_atr_mult",   0.2, 0.8),
        }

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr = df["atr"].ffill()
        tol = atr * p["retest_tolerance_atr"]

        # ── Step 1: Find Order Blocks ─────────────────────────────────────────
        bull_ob, bear_ob, bull_ob_high_raw, bull_ob_low_raw, bear_ob_high_raw, bear_ob_low_raw = self.detect_order_blocks(df, p["ob_lookback"], p["ob_body_pct"])

        bull_ob_high = bull_ob_high_raw.ffill()
        bull_ob_low  = bull_ob_low_raw.ffill()
        bear_ob_high = bear_ob_high_raw.ffill()
        bear_ob_low  = bear_ob_low_raw.ffill()

        # ── Step 2: Detect break-through (displacement) ───────────────────────
        # Displacement = bar range >= break_atr_mult * ATR
        displacement = df["range"] >= (atr * p["break_atr_mult"])

        # Bullish OB broken downward (price closes below OB low with displacement)
        # → becomes a Bearish Breaker Block
        bear_breaker = (
            (df["close"] < bull_ob_low) &
            displacement &
            df["bullish"].shift(1).fillna(False)  # prior bar was bullish (in OB)
        )

        # Bearish OB broken upward → Bullish Breaker Block
        bull_breaker = (
            (df["close"] > bear_ob_high) &
            displacement &
            (~df["bullish"].shift(1).fillna(True))
        )

        # Track breaker zone levels
        bull_breaker_low  = df["low"].where(bull_breaker).ffill()
        bull_breaker_high = df["high"].where(bull_breaker).ffill()
        bear_breaker_low  = df["low"].where(bear_breaker).ffill()
        bear_breaker_high = df["high"].where(bear_breaker).ffill()

        # Age of last breaker
        bull_breaker_age = self._bars_since(bull_breaker)
        bear_breaker_age = self._bars_since(bear_breaker)

        # ── Step 3: Retest of Breaker Block ───────────────────────────────────
        age = p["max_breaker_age_bars"]

        # Bullish retest: price dips back into bull breaker zone from above
        at_bull_breaker = (
            (bull_breaker_age > 0) & (bull_breaker_age <= age) &
            (df["low"]  <= bull_breaker_high + tol) &
            (df["close"] >= bull_breaker_low  - tol)
        )
        # Bearish retest: price rises back into bear breaker zone from below
        at_bear_breaker = (
            (bear_breaker_age > 0) & (bear_breaker_age <= age) &
            (df["high"] >= bear_breaker_low  - tol) &
            (df["close"] <= bear_breaker_high + tol)
        )

        # ── Step 4: Confluence ────────────────────────────────────────────────
        bull_fvg, bear_fvg = self.detect_fvg(df, p["fvg_min_atr_mult"])
        fvg_bull_window = self._rolling_any(bull_fvg, age)
        fvg_bear_window = self._rolling_any(bear_fvg, age)

        bias  = self.htf_bias(df)
        in_kz = self.kill_zone_mask(df, p.get("kill_zones", ["london", "ny_am"]))
        vol   = self.volume_filter(df)

        long_conf = (
            at_bull_breaker.astype(int) +        # [1] at breaker retest
            displacement.shift(1).fillna(False).astype(int) + # [2] prior displacement
            (bias == 1).astype(int) +             # [3] HTF bullish
            fvg_bull_window.astype(int) +         # [4] FVG in leg
            in_kz.astype(int) +                   # [5] kill zone
            vol.astype(int)                        # [6] volume
        )
        short_conf = (
            at_bear_breaker.astype(int) +
            displacement.shift(1).fillna(False).astype(int) +
            (bias == -1).astype(int) +
            fvg_bear_window.astype(int) +
            in_kz.astype(int) +
            vol.astype(int)
        )

        out["long_signal"]  = at_bull_breaker & (long_conf  >= self.min_confluence)
        out["short_signal"] = at_bear_breaker & (short_conf >= self.min_confluence)

        out.loc[out["long_signal"],  "entry_price"] = df.loc[out["long_signal"],  "close"]
        out.loc[out["short_signal"], "entry_price"] = df.loc[out["short_signal"], "close"]

        atr_sl = atr * p["atr_sl_mult"]
        out.loc[out["long_signal"],  "stop_loss"] = (
            bull_breaker_low.loc[out["long_signal"]] - atr_sl.loc[out["long_signal"]]
        )
        out.loc[out["short_signal"], "stop_loss"] = (
            bear_breaker_high.loc[out["short_signal"]] + atr_sl.loc[out["short_signal"]]
        )

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[out["long_signal"],  "take_profit"] = out.loc[out["long_signal"],  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[out["short_signal"], "take_profit"] = out.loc[out["short_signal"], "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(out["long_signal"],  long_conf,
                                  np.where(out["short_signal"], short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)

    @staticmethod
    def _bars_since(s: pd.Series) -> pd.Series:
        arr    = s.values.astype(bool)
        result = np.full(len(arr), 9999, dtype=int)
        last   = -9999
        for i, v in enumerate(arr):
            if v:
                last = i
            result[i] = i - last if last >= 0 else 9999
        return pd.Series(result, index=s.index)

    @staticmethod
    def _rolling_any(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=1).max().astype(bool)
