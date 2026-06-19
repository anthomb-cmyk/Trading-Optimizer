"""
Strategy — Opening Range Breakout (ORB) with volume confirmation
================================================================

Edge / evidence
---------------
The first few minutes of the US regular-session (RTH) open carry the day's
heaviest, most informed order flow.  The high/low established in that opening
window (the "opening range", OR) acts as an intraday reference level.  When
price subsequently breaks the OR boundary on EXPANDING volume, it tends to
continue in the breakout direction for the remainder of the session — the
classic Opening Range Breakout that has been documented since Crabel.  The
volume gate is what separates a genuine institutional break from a low-volume
fake-out.

Logic (per RTH session)
-----------------------
  1. Opening range (OR):  OR_high / OR_low = the high / low of the FIRST
     ``or_bars`` bars of the session.  The OR is KNOWN ONLY AFTER those
     ``or_bars`` bars have completed (causal: it is the cumulative high/low of
     bars 0..or_bars-1, frozen at the close of bar or_bars-1).

  2. Breakout window:  bars from session-position ``or_bars`` onward, within the
     SAME session.
       long  when close > OR_high  AND  volume > vol_mult * vol_ma
       short when close < OR_low   AND  volume > vol_mult * vol_ma

  3. Stop-loss:  the OPPOSITE side of the opening range
       long  stop = OR_low   - atr_sl_mult * atr  (buffer below the range)
       short stop = OR_high  + atr_sl_mult * atr  (buffer above the range)
     The OR boundary is the structural invalidation level; the
     ``atr_sl_mult * atr`` term adds a volatility buffer so the stop sits just
     beyond the range rather than exactly on it.  Set ``atr_sl_mult=0`` to put
     the stop exactly on the opposite OR boundary.

  4. Take-profit:  entry +/- sl_distance * rr_ratio  (fixed reward:risk).

  5. At most the FIRST breakout per side per session (the first long and the
     first short each fire at most once per session).

Confluence checklist
-------------------
  [1] Inside the breakout window of the configured RTH session   ← required
  [2] Close breaks the opening range boundary                    ← required
  [3] Volume expansion: volume > vol_mult * vol_ma               ← required
  [4] HTF bias (EMA-200) aligned with the breakout direction     (adds weight)

CAUSALITY
---------
Every session-anchored feature is built from WITHIN-SESSION data UP TO the
current bar only:
  * The bar-position within the session is a cumulative count (cumcount), never
    a full-session size.
  * OR_high / OR_low are cumulative max / min over the first ``or_bars`` bars,
    frozen at the OR-window close and forward-filled for the rest of the
    session — never a ``groupby.transform('max')`` full-session aggregate that
    would peek at later bars.
  * The "first breakout per side per session" gate uses a cumulative count of
    prior triggers within the session, so a bar's flag depends only on bars
    ``<= t``.
  * No negative shifts anywhere.  A breakout is confirmed at the bar whose
    CLOSE crosses the boundary — the bar at which it becomes known.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.settings import SESSIONS
from strategies.base_strategy import BaseStrategy
from strategies import regime as regime_mod


class OpeningRangeBreakout(BaseStrategy):
    name        = "opening_range_breakout"
    system_role = "primary"
    required_timeframes = ["15m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "or_bars":      2,            # bars defining the opening range
            "vol_mult":     1.3,          # volume must exceed vol_mult * vol_ma
            "atr_sl_mult":  1.0,          # ATR buffer beyond the opposite OR side
            "rr_ratio":     2.0,          # take-profit reward:risk
            "session":      "new_york",   # RTH session key (config.settings.SESSIONS)
            "require_bias": False,        # if True, HTF bias must align (4th confluence)
            "regime_mode":  "off",        # {"off","trend","range"} regime gate (causal)
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "or_bars":      trial.suggest_int("orb_or_bars",          1, 8),
            "vol_mult":     trial.suggest_float("orb_vol_mult",       1.0, 2.5),
            "atr_sl_mult":  trial.suggest_float("orb_atr_sl_mult",    0.0, 2.0),
            "rr_ratio":     trial.suggest_float("orb_rr_ratio",       1.5, 4.0),
            "session":      trial.suggest_categorical("orb_session",  ["new_york", "london"]),
            "require_bias": trial.suggest_categorical("orb_require_bias", [True, False]),
            "regime_mode":  trial.suggest_categorical("orb_regime_mode", ["off", "trend", "range"]),
        }

    # ── Session geometry helpers (all causal) ─────────────────────────────────

    @staticmethod
    def _time_to_float(t: str) -> float:
        h, m = t.split(":")
        return int(h) + int(m) / 60.0

    def _session_mask(self, df: pd.DataFrame, session: str) -> pd.Series:
        """True for bars inside *session* (UTC), mirrors data.loader.get_session_mask."""
        sess = SESSIONS.get(session)
        if sess is None:
            raise ValueError(f"Unknown session '{session}'. Valid: {list(SESSIONS)}")
        hour = df.index.hour + df.index.minute / 60.0
        start_h = self._time_to_float(sess["start"])
        end_h   = self._time_to_float(sess["end"])
        return pd.Series((hour >= start_h) & (hour < end_h), index=df.index)

    @staticmethod
    def _session_id(df: pd.DataFrame, in_session: pd.Series) -> pd.Series:
        """
        Stable per-session identifier for in-session bars, NaN otherwise.

        Causal: derived purely from the timestamp index (the UTC calendar date),
        which is known at each bar.  Two RTH sessions never share a UTC date
        (each session sits within one calendar day), so the date is a valid
        session key.  Out-of-session bars get NaN.
        """
        dates = pd.Series(df.index.normalize(), index=df.index)
        return dates.where(in_session)

    def _opening_range(
        self,
        df: pd.DataFrame,
        in_session: pd.Series,
        sess_id: pd.Series,
        or_bars: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Build the cumulative, CAUSAL opening range per session.

        Returns
        -------
        or_high, or_low : Series
            The opening-range high/low, known (non-NaN) only AT and AFTER the OR
            window has fully formed (session-position >= or_bars - 1 carries the
            cumulative value; from or_bars onward it is frozen).  NaN before the
            OR is complete and on out-of-session bars.
        pos_in_session : Series
            0-based bar position within the session (cumcount), NaN off-session.
        or_complete : Series
            True on bars where the OR window has fully formed AND we are strictly
            past it (pos_in_session >= or_bars), i.e. the breakout window.

        Causality
        ---------
        * ``pos_in_session`` is a cumulative count (groupby.cumcount), which only
          ever uses bars seen so far.
        * ``or_high``/``or_low`` are the cumulative max/min of the high/low over
          the FIRST ``or_bars`` bars of the session.  We compute the running
          max/min over the OR-window bars only, then forward-fill that frozen
          value across the rest of the session.  No future bar is referenced.
        """
        # 0-based position within each session. cumcount is purely backward-looking.
        grp = df.groupby(sess_id)
        pos = grp.cumcount().astype(float)
        pos = pos.where(in_session)  # NaN off-session

        # Highs/lows restricted to the OR window (first or_bars bars of session).
        in_or_window = in_session & (pos < or_bars)
        or_high_src = df["high"].where(in_or_window)
        or_low_src  = df["low"].where(in_or_window)

        # Running (cumulative) max/min WITHIN the session, over OR-window bars only.
        # cummax/cummin are causal: value at t uses only bars <= t.
        run_high = or_high_src.groupby(sess_id).cummax()
        run_low  = or_low_src.groupby(sess_id).cummin()

        # The OR value is final once we reach the last OR-window bar
        # (pos == or_bars - 1).  Freeze it there and forward-fill within session.
        or_high = run_high.groupby(sess_id).ffill()
        or_low  = run_low.groupby(sess_id).ffill()

        # Restrict the usable OR to bars where the OR window has fully closed.
        or_complete = in_session & (pos >= or_bars) & or_high.notna() & or_low.notna()

        return or_high, or_low, pos, or_complete.fillna(False)

    @staticmethod
    def _first_per_session(trigger: pd.Series, sess_id: pd.Series) -> pd.Series:
        """
        Keep only the FIRST True per session for *trigger*.

        Causal: uses a within-session cumulative count of triggers seen so far;
        a bar is the first trigger iff it is True and no earlier in-session bar
        was True.  Depends only on bars <= t.
        """
        trig = trigger.fillna(False).astype(int)
        # Number of triggers strictly BEFORE this bar, within the session.
        prior = trig.groupby(sess_id).cumsum() - trig
        return (trigger.fillna(False)) & (prior == 0)

    # ── Signal generation ─────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        or_bars = int(p["or_bars"])
        atr     = df["atr"].ffill()

        # ── Session geometry ──────────────────────────────────────────────────
        in_session = self._session_mask(df, p["session"])
        sess_id    = self._session_id(df, in_session)

        # ── Causal opening range ──────────────────────────────────────────────
        or_high, or_low, pos, or_complete = self._opening_range(
            df, in_session, sess_id, or_bars
        )

        # ── Volume gate (causal: vol_ma is a trailing rolling mean) ────────────
        if "vol_ma" in df.columns and not df["vol_ma"].isna().all():
            vol_ok = (df["volume"] > df["vol_ma"] * p["vol_mult"]).fillna(False)
        else:
            # No usable volume series → volume gate cannot confirm. Be strict
            # (the edge depends on volume) but do not crash.
            vol_ok = pd.Series(False, index=df.index)

        # ── Breakout detection (close beyond the OR boundary) ─────────────────
        long_break  = or_complete & (df["close"] > or_high) & vol_ok
        short_break = or_complete & (df["close"] < or_low)  & vol_ok

        # ── At most the first breakout per side per session ───────────────────
        long_trigger  = self._first_per_session(long_break,  sess_id)
        short_trigger = self._first_per_session(short_break, sess_id)

        # A session that broke up first should not also fire a (later) first
        # short on the SAME bar; long/short on one bar are mutually exclusive by
        # construction (close cannot be both > or_high and < or_low).

        # ── Confluence ────────────────────────────────────────────────────────
        bias = self.htf_bias(df)  # +1 / -1 / 0 from EMA-200 (causal)

        long_conf = (
            or_complete.astype(int) +                       # [1] in breakout window
            (df["close"] > or_high).fillna(False).astype(int) +  # [2] break of OR_high
            vol_ok.astype(int) +                            # [3] volume expansion
            (bias == 1).astype(int)                         # [4] HTF bullish
        )
        short_conf = (
            or_complete.astype(int) +
            (df["close"] < or_low).fillna(False).astype(int) +
            vol_ok.astype(int) +
            (bias == -1).astype(int)
        )

        # Optional hard bias requirement.
        if p.get("require_bias", False):
            long_trigger  = long_trigger  & (bias == 1)
            short_trigger = short_trigger & (bias == -1)

        # ── Regime gate (causal) ──────────────────────────────────────────────
        # Restrict triggers to favourable regimes using strategies.regime, whose
        # labels/gates are append-invariant (built from causal ADX + trailing ATR
        # percentile). "off" = no gating (preserves baseline behaviour);
        # "trend" = only trend-regime bars; "range" = only range-regime bars.
        regime_mode = str(p.get("regime_mode", "off"))
        if regime_mode in ("trend", "range"):
            labels = regime_mod.regime_label(df)
            regime_gate = labels.str.startswith(regime_mode).fillna(False)
            long_trigger  = long_trigger  & regime_gate
            short_trigger = short_trigger & regime_gate

        out["long_signal"]  = long_trigger
        out["short_signal"] = short_trigger

        # ── Entry / stop / target ─────────────────────────────────────────────
        out.loc[long_trigger,  "entry_price"] = df.loc[long_trigger,  "close"]
        out.loc[short_trigger, "entry_price"] = df.loc[short_trigger, "close"]

        atr_buf = atr * p["atr_sl_mult"]
        # Stop on the OPPOSITE side of the opening range, plus an ATR buffer.
        out.loc[long_trigger,  "stop_loss"] = or_low.loc[long_trigger]   - atr_buf.loc[long_trigger]
        out.loc[short_trigger, "stop_loss"] = or_high.loc[short_trigger] + atr_buf.loc[short_trigger]

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[long_trigger,  "take_profit"] = out.loc[long_trigger,  "entry_price"] + sl_dist * p["rr_ratio"]
        out.loc[short_trigger, "take_profit"] = out.loc[short_trigger, "entry_price"] - sl_dist * p["rr_ratio"]

        out["confluence_count"] = np.where(long_trigger,  long_conf,
                                  np.where(short_trigger, short_conf, 0))

        return self._finalize_signals(out, self.min_confluence)
