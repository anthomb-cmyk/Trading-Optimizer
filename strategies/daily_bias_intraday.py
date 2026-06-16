"""
Strategy 10 — Daily Bias + Intraday Entry

This is the MACRO FILTER for the entire APEX system. It determines the
directional bias for the trading day and provides a scored bias signal that
all other strategies consume via StrategySystem.

Bias is calculated from THREE independent sources. When 2 or 3 agree,
the bias is strong enough to trade. One or zero agreement = no trade.

Bias sources:
  1. Daily Open:   price above/below today's opening price
  2. Daily EMA:    price above/below the 20-period daily EMA (approx as 20×TF bars)
  3. Daily Structure: last two swing highs and lows form HH/HL (bull) or LL/LH (bear)

Bias score: 0–3 (number of agreeing sources). Threshold for trading = 2.

Intraday entry:
  When bias is set, this strategy also generates its OWN lower-TF entry signals
  using a market structure shift (CHoCH) aligned with the bias. These fire
  independently of the primary strategies and are combined by StrategySystem.

System interface:
  get_bias(df, params) → pd.Series of +1/0/-1 per bar (used by StrategySystem)
  generate_signals()   → standard signal columns for standalone backtesting
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class DailyBiasIntraday(BaseStrategy):
    name        = "daily_bias_intraday"
    system_role = "bias_filter"
    required_timeframes = ["1h", "15m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "daily_ema_bars":     20,           # approximate daily EMA period in entry-TF bars
            "bias_threshold":     2,            # sources that must agree (out of 3)
            "structure_lookback": 40,           # bars for swing structure analysis
            "entry_on_choch":     True,         # enter on CHoCH aligned with bias
            "entry_on_bos":       False,        # also enter on BOS (more aggressive)
            "fvg_min_atr_mult":   0.25,
            "ob_lookback":        15,
            "ob_body_pct":        0.55,
            "atr_sl_mult":        1.0,
            "rr_ratio":           2.5,
            "kill_zones":         ["london", "ny_am"],
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "daily_ema_bars":     trial.suggest_int("dbi_daily_ema_bars",     10, 40),
            "bias_threshold":     trial.suggest_int("dbi_bias_threshold",     1, 3),
            "structure_lookback": trial.suggest_int("dbi_structure_lookback", 20, 80),
            "entry_on_choch":     trial.suggest_categorical("dbi_on_choch", [True, False]),
            "entry_on_bos":       trial.suggest_categorical("dbi_on_bos",   [True, False]),
            "fvg_min_atr_mult":   trial.suggest_float("dbi_fvg_min_atr",    0.15, 0.50),
            "ob_lookback":        trial.suggest_int("dbi_ob_lookback",       8, 30),
            "ob_body_pct":        trial.suggest_float("dbi_ob_body_pct",     0.45, 0.80),
            "atr_sl_mult":        trial.suggest_float("dbi_atr_sl_mult",     0.6, 2.0),
            "rr_ratio":           trial.suggest_float("dbi_rr_ratio",        1.5, 4.0),
        }

    # ── System interface ──────────────────────────────────────────────────────

    def get_bias(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Returns +1 (bull), -1 (bear), 0 (mixed/no trade) per bar.
        This is the primary output consumed by StrategySystem.
        """
        p           = {**self.default_params, **params}
        score       = self._bias_score(df, p)
        threshold   = p["bias_threshold"]

        bias = np.where(score >=  threshold,  1,
               np.where(score <= -threshold, -1, 0))
        return pd.Series(bias, index=df.index)

    def get_bias_score(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Returns the raw score (−3 to +3) for analysis / reporting."""
        p = {**self.default_params, **params}
        return self._bias_score(df, p)

    # ── Main signal generation ────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr   = df["atr"].ffill()
        bias  = self.get_bias(df, p)
        in_kz = self.kill_zone_mask(df, p.get("kill_zones", ["london", "ny_am"]))

        # ── LTF entry triggers aligned with bias ──────────────────────────────
        bull_choch, bear_choch = self.detect_choch(df, p["structure_lookback"])
        bull_bos,   bear_bos   = self.detect_bos(df, p["structure_lookback"])

        bull_fvg, bear_fvg = self.detect_fvg(df, p["fvg_min_atr_mult"])
        fvg_bull_w = self._rolling_any(bull_fvg, 8)
        fvg_bear_w = self._rolling_any(bear_fvg, 8)

        bull_ob, bear_ob = self.detect_order_blocks(df, p["ob_lookback"], p["ob_body_pct"])
        ob_bull_w = self._rolling_any(bull_ob, p["ob_lookback"])
        ob_bear_w = self._rolling_any(bear_ob, p["ob_lookback"])

        # Entry trigger: CHoCH and/or BOS aligned with bias
        long_trigger  = (bias == 1) & in_kz & (
            (bull_choch if p["entry_on_choch"] else pd.Series(False, index=df.index)) |
            (bull_bos   if p["entry_on_bos"]   else pd.Series(False, index=df.index))
        )
        short_trigger = (bias == -1) & in_kz & (
            (bear_choch if p["entry_on_choch"] else pd.Series(False, index=df.index)) |
            (bear_bos   if p["entry_on_bos"]   else pd.Series(False, index=df.index))
        )

        # Confluence: bias strength + supporting structure
        bias_score    = self._bias_score(df, p)
        long_conf = (
            (bias == 1).astype(int) * 2 +         # [1+2] strong bias (counts as 2)
            long_trigger.astype(int) +             # [3] LTF trigger
            (fvg_bull_w | ob_bull_w).astype(int) + # [4] FVG or OB
            in_kz.astype(int) +                    # [5] kill zone
            self.volume_filter(df).astype(int)     # [6] volume
        )
        short_conf = (
            (bias == -1).astype(int) * 2 +
            short_trigger.astype(int) +
            (fvg_bear_w | ob_bear_w).astype(int) +
            in_kz.astype(int) +
            self.volume_filter(df).astype(int)
        )

        out["long_signal"]  = long_trigger  & (long_conf  >= self.min_confluence)
        out["short_signal"] = short_trigger & (short_conf >= self.min_confluence)

        out.loc[out["long_signal"],  "entry_price"] = df.loc[out["long_signal"],  "close"]
        out.loc[out["short_signal"], "entry_price"] = df.loc[out["short_signal"], "close"]

        atr_sl = atr * p["atr_sl_mult"]
        out.loc[out["long_signal"],  "stop_loss"] = df.loc[out["long_signal"],  "low"]  - atr_sl.loc[out["long_signal"]]
        out.loc[out["short_signal"], "stop_loss"] = df.loc[out["short_signal"], "high"] + atr_sl.loc[out["short_signal"]]

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[out["long_signal"],  "take_profit"] = out.loc[out["long_signal"],  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[out["short_signal"], "take_profit"] = out.loc[out["short_signal"], "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(out["long_signal"],  long_conf,
                                  np.where(out["short_signal"], short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)

    # ── Bias calculation ──────────────────────────────────────────────────────

    def _bias_score(self, df: pd.DataFrame, p: Dict[str, Any]) -> pd.Series:
        """
        Returns integer score in range [−3, +3].
        Each of the 3 bias sources contributes +1 (bull) or −1 (bear).
        """
        # Source 1: Daily open
        daily_open = (
            df.groupby(df.index.date)["open"]
            .transform("first")
        )
        daily_open.index = df.index
        s1 = np.where(df["close"] > daily_open,  1,
             np.where(df["close"] < daily_open, -1, 0))

        # Source 2: Daily EMA (approximated as slow EMA on entry timeframe)
        daily_ema = df["close"].ewm(span=p["daily_ema_bars"], adjust=False).mean()
        s2 = np.where(df["close"] > daily_ema,  1,
             np.where(df["close"] < daily_ema, -1, 0))

        # Source 3: Swing structure — last swing high and swing low direction
        lb             = p["structure_lookback"]
        last_sh        = df["high"].where(df["swing_high"]).ffill()
        last_sl        = df["low"].where(df["swing_low"]).ffill()
        prev_sh        = df["high"].where(df["swing_high"]).shift(lb).ffill()
        prev_sl        = df["low"].where(df["swing_low"]).shift(lb).ffill()

        hh = last_sh > prev_sh   # higher high
        hl = last_sl > prev_sl   # higher low
        ll = last_sl < prev_sl   # lower low
        lh = last_sh < prev_sh   # lower high

        s3 = np.where(hh & hl,  1,
             np.where(ll & lh, -1, 0))

        score = pd.Series(s1 + s2 + s3, index=df.index)
        return score
