"""
Strategy — Session-Anchored VWAP Mean Reversion

Edge (evidence-based):
  Intraday index futures (ES/MES, NQ/MNQ) mean-revert toward the session
  Volume-Weighted Average Price (VWAP) during the regular trading session.
  VWAP is the institutional benchmark execution price for the day; when price
  stretches a statistically large distance away from it, passive flow tends to
  pull price back toward fair value.  The robust, tradeable version of this is
  NOT "fade any deviation" (that bleeds in trends) but: wait for price to be a
  dispersion-band multiple away from session VWAP AND for an actual reversal
  candle to print (a bar that closes back toward VWAP with a dominant body in
  the reversion direction).  The reversal candle is the bar at which the
  setup becomes confirmable from past data only — entries fire on that bar.

  Steps:
    1. Compute a CAUSAL session-anchored VWAP that RESETS at the start of each
       RTH session: vwap_t = cumsum(typical*vol)/cumsum(vol) over bars in the
       current session up to and including bar t.  typical = (h + l + c) / 3.
       No within-session aggregate ever uses a future bar.
    2. Compute a causal dispersion band: rolling std of (close - vwap) over a
       short window (falls back to ATR when std is unavailable).  The band
       width at bar t uses only bars <= t.
    3. Long when close is band_mult bands BELOW VWAP and a BULLISH reversal
       candle prints (close > open, body >= reversal_body_pct of range, closes
       back up toward VWAP).  Short is the mirror.
    4. Target = VWAP itself (the reversion magnet), but never closer than the
       rr_ratio floor versus the stop.  Stop = atr_sl_mult * ATR beyond the
       bar extreme (low for longs, high for shorts).

Confluence checklist:
  [1] Inside the configured RTH session (within-session bar)          ← required
  [2] Price stretched band_mult dispersion bands beyond session VWAP  ← required
  [3] Reversal candle in the reversion direction (body filter)        ← required
  [4] HTF bias not fighting the reversion (allow neutral)
  [5] RSI confirms exhaustion (oversold for longs / overbought shorts)
  [6] Volume expansion on the reversal bar (capitulation / absorption)

CAUSALITY:
  Every series used at bar t depends only on indices <= t.  The session VWAP is
  built with a per-session cumulative sum (group key = session id that is
  itself causal: it increments on the first bar of each session), so it is
  append-invariant: extending the frame with future bars never alters an
  already-emitted VWAP, band, signal, or level.  No negative shifts anywhere.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.settings import SESSIONS
from strategies.base_strategy import BaseStrategy
from strategies import regime


class VWAPReversion(BaseStrategy):
    name        = "vwap_reversion"
    system_role = "primary"
    required_timeframes = ["5m", "15m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "band_mult":         2.0,    # dispersion bands away from VWAP to qualify
            "atr_sl_mult":       1.0,    # ATR multiple beyond the extreme for the stop
            "rr_ratio":          1.5,    # minimum reward:risk floor when VWAP target is tight
            "reversal_body_pct": 0.5,    # reversal candle body must be >= x * range
            "session":           "new_york",  # RTH session key from SESSIONS
            "disp_window":       20,     # rolling window for VWAP-deviation dispersion (std)
            "rsi_os":            35,     # RSI oversold threshold (long confirmation)
            "rsi_ob":            65,     # RSI overbought threshold (short confirmation)
            "regime_mode":       "off",  # regime gate: "off" | "trend" | "range" (default off = current behavior)
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        # UNIQUE Optuna keys prefixed with "vwr_" to avoid collisions.
        return {
            "band_mult":         trial.suggest_float("vwr_band_mult",        1.0, 3.5),
            "atr_sl_mult":       trial.suggest_float("vwr_atr_sl_mult",      0.5, 2.0),
            "rr_ratio":          trial.suggest_float("vwr_rr_ratio",         1.0, 3.0),
            "reversal_body_pct": trial.suggest_float("vwr_reversal_body",    0.35, 0.75),
            "session":           trial.suggest_categorical("vwr_session",
                                                           ["new_york", "london"]),
            "disp_window":       trial.suggest_int("vwr_disp_window",        10, 40),
            "rsi_os":            trial.suggest_int("vwr_rsi_os",             20, 45),
            "rsi_ob":            trial.suggest_int("vwr_rsi_ob",             55, 80),
            "regime_mode":       trial.suggest_categorical("vwr_regime_mode",
                                                           ["off", "trend", "range"]),
        }

    # ── Causal session-anchored VWAP ──────────────────────────────────────────

    @staticmethod
    def _session_mask(df: pd.DataFrame, session: str) -> pd.Series:
        """True for bars inside *session* (UTC), per SESSIONS in settings."""
        sess = SESSIONS.get(session)
        if sess is None:
            raise ValueError(f"Unknown session '{session}'")

        def _tf(t: str) -> float:
            h, m = t.split(":")
            return int(h) + int(m) / 60.0

        hour    = df.index.hour + df.index.minute / 60.0
        start_h = _tf(sess["start"])
        end_h   = _tf(sess["end"])
        return pd.Series((hour >= start_h) & (hour < end_h), index=df.index)

    @staticmethod
    def _session_id(in_session: pd.Series) -> pd.Series:
        """
        Causal session-instance id.  Increments by 1 on the FIRST in-session bar
        of every new session block (a False→True transition).  Out-of-session
        bars carry the id of the session they belong to is irrelevant — they are
        masked out downstream — but they keep the running id so the cumulative
        reset is well defined.  Depends only on bars <= t (it is a forward scan).
        """
        is_first = in_session & ~in_session.shift(1, fill_value=False)
        return is_first.cumsum()

    @classmethod
    def _session_vwap(
        cls,
        df: pd.DataFrame,
        in_session: pd.Series,
    ) -> pd.Series:
        """
        Cumulative VWAP that RESETS each session.  Strictly causal:
          vwap_t = (Σ typical_i * vol_i) / (Σ vol_i)   for i in current session, i <= t.
        Out-of-session bars (and any leading bars before the first session)
        are NaN.  No future bar can influence vwap_t.
        """
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vol     = df["volume"].astype(float)

        sid = cls._session_id(in_session)

        # Per-session cumulative sums.  cumsum within a groupby is order-preserving
        # and causal: position t only sums positions <= t that share the group id.
        pv = (typical * vol).where(in_session, 0.0)
        vv = vol.where(in_session, 0.0)

        cum_pv = pv.groupby(sid).cumsum()
        cum_vv = vv.groupby(sid).cumsum()

        vwap = cum_pv / cum_vv.replace(0.0, np.nan)
        # Only meaningful inside a session.
        return vwap.where(in_session)

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        atr        = df["atr"].ffill()
        in_session = self._session_mask(df, p["session"])

        # ── Step 1: causal session-anchored VWAP ──────────────────────────────
        vwap = self._session_vwap(df, in_session)
        dev  = df["close"] - vwap          # signed deviation from fair value

        # ── Step 2: causal dispersion band ────────────────────────────────────
        # Rolling std of the VWAP deviation; window uses only bars <= t.
        # min_periods kept modest so bands form early in the session; where the
        # std is unavailable/zero we fall back to ATR so the band never collapses.
        disp = (
            dev.rolling(p["disp_window"], min_periods=5)
               .std()
        )
        band = disp.where(disp > 0).fillna(atr)
        band = band.where(band > 0, atr)

        # ── Step 3: reversal candle (body-dominant, closes toward VWAP) ───────
        rng      = df["range"].replace(0, np.nan)
        body_pct = (df["body"] / rng).fillna(0.0)
        strong_body = body_pct >= p["reversal_body_pct"]

        bull_reversal = strong_body & df["bullish"]            # close > open
        bear_reversal = strong_body & ~df["bullish"]           # close < open

        # ── Step 4: stretch beyond VWAP by band_mult dispersion bands ─────────
        below_band = dev <= -(p["band_mult"] * band)           # price under VWAP
        above_band = dev >=  (p["band_mult"] * band)           # price over VWAP

        # ── Confluence ────────────────────────────────────────────────────────
        bias = self.htf_bias(df)                               # +1 / -1 / 0
        rsi  = df.get("rsi", pd.Series(50.0, index=df.index)).fillna(50.0)
        vol  = self.volume_filter(df, 1.0)

        rsi_long  = rsi <= p["rsi_os"]
        rsi_short = rsi >= p["rsi_ob"]

        long_conf = (
            in_session.astype(int) +          # [1] inside RTH session
            below_band.astype(int) +          # [2] stretched below VWAP
            bull_reversal.astype(int) +       # [3] bullish reversal candle
            (bias != -1).astype(int) +        # [4] HTF not bearish
            rsi_long.astype(int) +            # [5] RSI oversold
            vol.astype(int)                   # [6] volume expansion
        )
        short_conf = (
            in_session.astype(int) +
            above_band.astype(int) +
            bear_reversal.astype(int) +
            (bias != 1).astype(int) +
            rsi_short.astype(int) +
            vol.astype(int)
        )

        # ── Regime gate (causal) ──────────────────────────────────────────────
        # Optional regime restriction.  regime.regime_label() is built solely
        # from causal helpers (adx / atr_percentile), so gating on it never
        # introduces look-ahead.  "off" leaves the original behavior untouched.
        #   "trend" -> only bars whose regime label is a trend regime
        #   "range" -> only bars whose regime label is a range regime
        regime_mode = p["regime_mode"]
        if regime_mode in ("trend", "range"):
            labels = regime.regime_label(df)
            regime_gate = labels.str.contains(regime_mode, na=False)
        else:
            regime_gate = pd.Series(True, index=df.index)

        # ── Triggers (all required legs present; confluence gate added too) ───
        long_trigger = (
            in_session & below_band & bull_reversal &
            vwap.notna() & regime_gate &
            (long_conf >= self.min_confluence)
        )
        short_trigger = (
            in_session & above_band & bear_reversal &
            vwap.notna() & regime_gate &
            (short_conf >= self.min_confluence)
        )

        out["long_signal"]  = long_trigger
        out["short_signal"] = short_trigger

        # ── Entry / stop / target ─────────────────────────────────────────────
        out.loc[long_trigger,  "entry_price"] = df.loc[long_trigger,  "close"]
        out.loc[short_trigger, "entry_price"] = df.loc[short_trigger, "close"]

        atr_sl = atr * p["atr_sl_mult"]
        # Stop beyond the reversal bar's extreme.
        out.loc[long_trigger,  "stop_loss"] = df.loc[long_trigger,  "low"]  - atr_sl.loc[long_trigger]
        out.loc[short_trigger, "stop_loss"] = df.loc[short_trigger, "high"] + atr_sl.loc[short_trigger]

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()

        # Primary target = session VWAP (the reversion magnet).  Enforce an
        # rr_ratio floor so a VWAP that sits too close to entry does not produce
        # a sub-threshold reward:risk trade.
        rr_long_tp  = out["entry_price"] + sl_dist * p["rr_ratio"]
        rr_short_tp = out["entry_price"] - sl_dist * p["rr_ratio"]

        # For longs, take the farther of VWAP and the rr floor (price is below
        # VWAP, so VWAP > entry; we want at least the rr floor above entry).
        out.loc[long_trigger,  "take_profit"] = np.maximum(
            vwap.loc[long_trigger], rr_long_tp.loc[long_trigger]
        )
        # For shorts, take the lower of VWAP and the rr floor.
        out.loc[short_trigger, "take_profit"] = np.minimum(
            vwap.loc[short_trigger], rr_short_tp.loc[short_trigger]
        )

        out["confluence_count"] = np.where(long_trigger,  long_conf,
                                  np.where(short_trigger, short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)
