"""
Strategy 4 — Market Structure Shift Entry

Logic:
  Tracks the fractal swing structure to identify when the prevailing trend
  changes direction (CHoCH) and when the new trend is confirmed (BOS).

  Trend identification:
    Bullish:  series of Higher Highs (HH) and Higher Lows (HL)
    Bearish:  series of Lower Lows  (LL) and Lower Highs (LH)

  CHoCH (Change of Character): first break of the last swing against the trend.
    - In a downtrend: closes above the last Lower High → potential reversal.
    - In an uptrend:  closes below the last Higher Low → potential reversal.

  BOS (Break of Structure): confirms the new direction.
    - After a bullish CHoCH: closes above the last swing high.
    - After a bearish CHoCH: closes below the last swing low.

  Entry modes:
    'aggressive' — enter on the CHoCH bar (early, tighter SL)
    'conservative' — wait for BOS + pullback to the CHoCH FVG

Confluence checklist:
  [1] CHoCH confirmed (closes through last swing)              ← required
  [2] BOS confirmed (conservative) or displacement candle (aggressive)
  [3] HTF bias aligns
  [4] FVG formed on the CHoCH / BOS candle
  [5] Inside kill zone
  [6] RSI divergence (price made new low/high but RSI didn't)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class MarketStructureShiftEntry(BaseStrategy):
    name        = "market_structure_shift"
    system_role = "confirmation"
    required_timeframes = ["15m", "1h"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "structure_lookback":    50,
            "choch_min_atr_mult":    0.5,
            "entry_type":            "aggressive",  # 'aggressive' | 'conservative'
            "pullback_bars":         5,              # bars to wait for pullback (conservative)
            "atr_sl_mult":           1.2,
            "rr_ratio":              2.5,
            "kill_zones":            ["london", "ny_am"],
            "fvg_min_atr_mult":      0.3,
            "rsi_divergence_window": 14,
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "structure_lookback": trial.suggest_int("mss_structure_lookback",   20, 100),
            "choch_min_atr_mult": trial.suggest_float("mss_choch_min_atr_mult", 0.3, 2.0),
            "entry_type":         trial.suggest_categorical("mss_entry_type", ["aggressive", "conservative"]),
            "pullback_bars":      trial.suggest_int("mss_pullback_bars",        2, 10),
            "atr_sl_mult":        trial.suggest_float("mss_atr_sl_mult",        0.6, 2.0),
            "rr_ratio":           trial.suggest_float("mss_rr_ratio",           1.5, 4.0),
            "fvg_min_atr_mult":   trial.suggest_float("mss_fvg_min_atr_mult",   0.2, 0.8),
        }

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr = df["atr"].ffill()
        lb  = p["structure_lookback"]

        # ── Step 1: CHoCH & BOS ───────────────────────────────────────────────
        bull_choch, bear_choch = self.detect_choch(df, lb)
        bull_bos,   bear_bos   = self.detect_bos(df, lb, confirm_bars=1)

        # Require CHoCH move is substantial (not just noise)
        min_move = atr * p["choch_min_atr_mult"]
        last_sh   = df["swing_high_price"].ffill()
        last_sl   = df["swing_low_price"].ffill()
        bull_choch = bull_choch & ((df["close"] - last_sh.shift(1)) >= min_move)
        bear_choch = bear_choch & ((last_sl.shift(1) - df["close"]) >= min_move)

        # ── Step 2: Entry trigger ─────────────────────────────────────────────
        if p["entry_type"] == "aggressive":
            long_trigger  = bull_choch
            short_trigger = bear_choch
        else:
            # Conservative: CHoCH occurred within pullback_bars, and now BOS confirmed
            bull_choch_recent = self._rolling_any(bull_choch, p["pullback_bars"])
            bear_choch_recent = self._rolling_any(bear_choch, p["pullback_bars"])
            long_trigger  = bull_bos & bull_choch_recent & ~bull_choch
            short_trigger = bear_bos & bear_choch_recent & ~bear_choch

        # ── Step 3: Confluence ────────────────────────────────────────────────
        bull_fvg, bear_fvg = self.detect_fvg(df, p["fvg_min_atr_mult"])
        fvg_bull_w = self._rolling_any(bull_fvg, p["pullback_bars"] + 2)
        fvg_bear_w = self._rolling_any(bear_fvg, p["pullback_bars"] + 2)

        bias  = self.htf_bias(df)
        in_kz = self.kill_zone_mask(df, p.get("kill_zones", ["london", "ny_am"]))
        vol   = self.volume_filter(df)

        # RSI divergence: price lower low but RSI higher low (bullish div)
        # Use rsi_divergence_window (default 14), not structure_lookback (50).
        rsi_win   = p["rsi_divergence_window"]
        rsi_vals  = df.get("rsi", pd.Series(50, index=df.index)).fillna(50)
        price_new_low  = df["close"] < df["close"].rolling(rsi_win).min().shift(1)
        rsi_not_low    = rsi_vals > rsi_vals.rolling(rsi_win).min().shift(1)
        bull_div       = price_new_low & rsi_not_low

        price_new_high = df["close"] > df["close"].rolling(rsi_win).max().shift(1)
        rsi_not_high   = rsi_vals < rsi_vals.rolling(rsi_win).max().shift(1)
        bear_div       = price_new_high & rsi_not_high

        long_conf = (
            long_trigger.astype(int) +          # [1] CHoCH / BOS
            bull_bos.astype(int) +              # [2] BOS present
            (bias == 1).astype(int) +           # [3] HTF bullish
            fvg_bull_w.astype(int) +            # [4] FVG
            in_kz.astype(int) +                 # [5] kill zone
            bull_div.astype(int)                # [6] RSI divergence
        )
        short_conf = (
            short_trigger.astype(int) +
            bear_bos.astype(int) +
            (bias == -1).astype(int) +
            fvg_bear_w.astype(int) +
            in_kz.astype(int) +
            bear_div.astype(int)
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
