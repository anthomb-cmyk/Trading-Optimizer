"""
Strategy — Short-Term Reversal (single-instrument mean reversion)

Edge (evidence-based):
  Sharp, fast directional moves in an intraday futures contract tend to
  over-extend.  When price travels much further than its recent volatility
  (ATR) would justify over a handful of bars, AND momentum (RSI) is at an
  exhaustion extreme, AND price has stretched away from its short-term mean
  (EMA-20), the move frequently snaps back toward the mean.  We FADE the move:

    Long  : recent return is sharply DOWN (<= -move_atr_mult * ATR)
            AND rsi < rsi_long (oversold)
            AND close stretched BELOW ema_20      -> buy the dip
    Short : recent return is sharply UP   (>= +move_atr_mult * ATR)
            AND rsi > rsi_short (overbought)
            AND close stretched ABOVE ema_20      -> sell the rip

  Stop  = atr_sl_mult * ATR beyond the move EXTREME (recent low for longs,
          recent high for shorts).  Take-profit = rr_ratio * stop distance,
          which targets reversion back toward the EMA-20 / mean.

CAUSALITY
---------
Every input at bar t uses ONLY data at indices <= t:
  * lookback_bars return  = close[t] - close[t - lookback_bars]   (past shift)
  * move extreme (low/high) = rolling min/max over the window ending at t
  * atr, rsi, ema_20      = pre-computed causal indicator columns (no -ve shift)
  * session mask          = pure function of each bar's own timestamp
The signal is confirmed and emitted at the SAME bar it becomes known (t),
mirroring the project pattern.  No negative shifts, no full-session
aggregates that peek ahead.

Confluence checklist (each adds 1):
  [1] Sharp move of the required magnitude in the faded direction   ← required
  [2] RSI at the exhaustion extreme
  [3] Price stretched the correct side of EMA-20
  [4] Inside the regular trading-hours session window
  [5] Volatility within acceptable bounds (not a vol blow-out)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class ShortTermReversal(BaseStrategy):
    name        = "short_term_reversal"
    system_role = "primary"
    required_timeframes = ["5m", "15m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "lookback_bars":  4,      # bars over which to measure the sharp move
            "move_atr_mult":  2.0,    # move must exceed this * ATR to qualify
            "rsi_long":       30,     # rsi below this = oversold (fade down -> long)
            "rsi_short":      70,     # rsi above this = overbought (fade up -> short)
            "atr_sl_mult":    1.0,    # stop = this * ATR beyond the move extreme
            "rr_ratio":       1.5,    # take-profit = rr_ratio * stop distance
            "stretch_atr_mult": 0.0,  # min distance of close from ema_20 in ATRs
            "session":        "new_york",  # RTH session window (see SESSIONS)
            "vol_max_mult":   3.0,    # skip when ATR > this * avg ATR (vol blow-out)
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "lookback_bars":    trial.suggest_int("str_lookback_bars",      2, 10),
            "move_atr_mult":    trial.suggest_float("str_move_atr_mult",    1.0, 4.0),
            "rsi_long":         trial.suggest_int("str_rsi_long",           15, 40),
            "rsi_short":        trial.suggest_int("str_rsi_short",          60, 85),
            "atr_sl_mult":      trial.suggest_float("str_atr_sl_mult",      0.5, 2.0),
            "rr_ratio":         trial.suggest_float("str_rr_ratio",         1.0, 3.0),
            "stretch_atr_mult": trial.suggest_float("str_stretch_atr_mult", 0.0, 1.5),
            "session":          trial.suggest_categorical(
                "str_session", ["new_york", "london", "asian"]
            ),
            "vol_max_mult":     trial.suggest_float("str_vol_max_mult",     2.0, 5.0),
        }

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        lb  = int(p["lookback_bars"])
        atr = df["atr"].ffill()

        # ── Recent move over the lookback window (causal) ─────────────────────
        # ret[t] = close[t] - close[t - lb]  → uses only indices <= t.
        ret = df["close"] - df["close"].shift(lb)
        move_thresh = atr * p["move_atr_mult"]

        # ── Move extremes within the lookback window ending at t (causal) ─────
        # rolling(lb+1) covers bars [t-lb .. t]; min_periods=lb+1 keeps it from
        # using partial (look-ahead-free but undefined) windows at the start.
        win = lb + 1
        recent_low  = df["low"].rolling(win, min_periods=win).min()
        recent_high = df["high"].rolling(win, min_periods=win).max()

        # ── Mean reference (EMA-20) and stretch (causal indicator) ────────────
        ema = df["ema_20"]
        stretch = atr * p["stretch_atr_mult"]
        # Long  needs close stretched BELOW the mean; short needs it ABOVE.
        below_mean = df["close"] < (ema - stretch)
        above_mean = df["close"] > (ema + stretch)

        # ── Momentum extreme (causal RSI) ─────────────────────────────────────
        rsi = df.get("rsi", pd.Series(50.0, index=df.index)).ffill().fillna(50.0)
        rsi_oversold   = rsi < p["rsi_long"]
        rsi_overbought = rsi > p["rsi_short"]

        # ── Session anchoring: only trade inside the RTH window ────────────────
        # Pure per-bar timestamp function — no cross-bar aggregation.
        in_session = self._session_mask(df, p["session"])

        # ── Volatility regime filter (causal; uses ffilled atr/atr_avg) ───────
        vol_ok = self.volatility_filter(df, max_mult=p["vol_max_mult"], min_mult=0.0)

        # ── Trigger conditions ────────────────────────────────────────────────
        sharp_down = ret <= -move_thresh                 # [1] long side
        sharp_up   = ret >=  move_thresh                 # [1] short side

        valid_extreme = (
            recent_low.notna() & recent_high.notna() &
            atr.notna() & (atr > 0) & ema.notna()
        )

        long_core  = sharp_down & rsi_oversold   & below_mean
        short_core = sharp_up   & rsi_overbought & above_mean

        long_trigger  = (long_core  & in_session & vol_ok & valid_extreme).fillna(False)
        short_trigger = (short_core & in_session & vol_ok & valid_extreme).fillna(False)

        # ── Confluence count (computed for every bar; gates validity later) ───
        long_conf = (
            sharp_down.fillna(False).astype(int) +   # [1]
            rsi_oversold.astype(int) +               # [2]
            below_mean.fillna(False).astype(int) +   # [3]
            in_session.astype(int) +                 # [4]
            vol_ok.astype(int)                       # [5]
        )
        short_conf = (
            sharp_up.fillna(False).astype(int) +
            rsi_overbought.astype(int) +
            above_mean.fillna(False).astype(int) +
            in_session.astype(int) +
            vol_ok.astype(int)
        )

        # ── Emit signals / prices ─────────────────────────────────────────────
        out["long_signal"]  = long_trigger
        out["short_signal"] = short_trigger

        out.loc[long_trigger,  "entry_price"] = df.loc[long_trigger,  "close"]
        out.loc[short_trigger, "entry_price"] = df.loc[short_trigger, "close"]

        atr_sl = atr * p["atr_sl_mult"]
        # Stop sits beyond the move EXTREME of the lookback window.
        out.loc[long_trigger,  "stop_loss"] = (
            recent_low.loc[long_trigger]   - atr_sl.loc[long_trigger]
        )
        out.loc[short_trigger, "stop_loss"] = (
            recent_high.loc[short_trigger] + atr_sl.loc[short_trigger]
        )

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[long_trigger,  "take_profit"] = (
            out.loc[long_trigger,  "entry_price"] + sl_dist.loc[long_trigger]  * p["rr_ratio"]
        )
        out.loc[short_trigger, "take_profit"] = (
            out.loc[short_trigger, "entry_price"] - sl_dist.loc[short_trigger] * p["rr_ratio"]
        )

        out["confluence_count"] = np.where(
            long_trigger,  long_conf,
            np.where(short_trigger, short_conf, 0),
        )

        return self._finalize_signals(out, self.min_confluence)

    # ── Session helper ────────────────────────────────────────────────────────
    @staticmethod
    def _session_mask(df: pd.DataFrame, session: str) -> pd.Series:
        """
        True for bars inside *session* (UTC), based solely on each bar's own
        timestamp.  Purely causal: no cross-bar aggregation, no peeking.
        Falls back to all-True if the index is not a DatetimeIndex.
        """
        from config.settings import SESSIONS

        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return pd.Series(True, index=idx)

        sess = SESSIONS.get(session)
        if sess is None:
            return pd.Series(True, index=idx)

        def _tf(t: str) -> float:
            h, m = t.split(":")
            return int(h) + int(m) / 60.0

        hour    = idx.hour + idx.minute / 60.0
        start_h = _tf(sess["start"])
        end_h   = _tf(sess["end"])
        return pd.Series((hour >= start_h) & (hour < end_h), index=idx)
