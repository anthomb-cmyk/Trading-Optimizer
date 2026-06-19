"""
Strategy — RTH Opening-Gap Fade  (gap_fade)

Logic:
  Index futures (ES/MES) frequently open the New York regular-trading-hours
  (RTH) session away from where the prior RTH session closed.  When that
  opening gap is large relative to volatility it tends to "fill" — price
  drifts back toward the prior RTH close intraday.  This strategy fades the
  gap:

    gap = session_open - prior_session_RTH_close            (both causal)

    gap up   (gap > 0, |gap| > gap_min_atr*ATR) → SHORT toward prior close
    gap down (gap < 0, |gap| > gap_min_atr*ATR) → LONG  toward prior close

  Entry fires on the FIRST bar of the RTH session (the open bar — the bar at
  which the gap first becomes known).  Stop is placed atr_sl_mult*ATR beyond
  the session extreme *since the open up to the current bar* (a causal,
  cumulative extreme — never the full-session high/low which would peek
  ahead).  On the open bar that extreme is just the open bar's own high/low.

Take-profit (documented):
  TARGET = the prior RTH close (the gap-fill level).  We additionally clamp
  it to be no closer than rr_ratio * stop-distance so a small gap still
  carries a sane reward:risk; whichever of {gap-fill level, rr_ratio target}
  is FARTHER from entry in the trade direction is used.  This makes the
  gap-fill the primary objective while guaranteeing a minimum rr_ratio of
  payoff geometry.  Exit by max_hold_bars if the gap is unfilled (handled by
  the backtest engine via a max-hold; documented here for completeness).

CAUSALITY (mandatory):
  * prior RTH close   — close of the LAST bar of the previous RTH session,
                        forward-filled to the current session.  Fully known
                        before the current session opens.
  * session open      — open of the FIRST RTH bar of the current session;
                        known at that bar.
  * gap               — derived from the two values above; known from the
                        session-open bar onward.
  * session extreme   — cumulative max(high)/min(low) of the current session
                        from its open UP TO the current bar (cummax/cummin
                        within the session group), never a full-session
                        aggregate.
  No negative shifts are used anywhere.

Confluence checklist:
  [1] Inside the RTH session                                  ← required
  [2] Gap magnitude exceeds gap_min_atr * ATR                 ← required
  [3] On the session-open (or first-confirmation) bar         ← required (entry)
  [4] HTF bias aligned with the fade direction
  [5] RSI not already stretched against the fade
  [6] Volatility within acceptable bounds

Optuna param keys are prefixed `gf_` to stay unique across strategies.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from config.settings import SESSIONS
from strategies.base_strategy import BaseStrategy


class GapFade(BaseStrategy):
    name        = "gap_fade"
    system_role = "primary"
    required_timeframes = ["5m", "15m"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "gap_min_atr":   0.5,    # gap must exceed x * ATR to be tradeable
            "atr_sl_mult":   1.0,    # stop = x * ATR beyond session extreme since open
            "rr_ratio":      1.5,    # minimum reward:risk floor for the target
            "max_hold_bars": 20,     # exit if unfilled by this many bars (engine-level)
            "session":       "new_york",
            "confirm_bars":  0,      # 0 = enter on open bar; N = wait N bars for confirmation
            "use_bias":      True,   # require HTF bias alignment for confluence weight
        }

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "gap_min_atr":   trial.suggest_float("gf_gap_min_atr",   0.2, 2.0),
            "atr_sl_mult":   trial.suggest_float("gf_atr_sl_mult",   0.5, 2.5),
            "rr_ratio":      trial.suggest_float("gf_rr_ratio",      1.0, 3.0),
            "max_hold_bars": trial.suggest_int("gf_max_hold_bars",   8,  40),
            "confirm_bars":  trial.suggest_int("gf_confirm_bars",    0,   3),
            "use_bias":      trial.suggest_categorical("gf_use_bias", [True, False]),
        }

    # ── Signal generation ──────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        if len(df) == 0:
            return self._finalize_signals(out, self.min_confluence)

        atr = df["atr"].ffill()

        # ── Step 1: RTH session mask ───────────────────────────────────────────
        in_rth = self._session_mask(df, p["session"])

        # ── Step 2: Per-session structure (all causal) ─────────────────────────
        # A "session id" increments each time we cross from outside RTH into RTH,
        # so each RTH block has its own id.  Bars outside RTH get id = NaN.
        sess_open, sess_id, bar_in_sess = self._session_open_and_index(df, in_rth)

        # Prior RTH close: the close of the last bar of the most recent COMPLETED
        # RTH session.  We capture each RTH session's last close as we leave the
        # session, then forward-fill it — so when a new session opens, the value
        # carried is the previous session's final close.  Fully causal.
        prior_close = self._prior_rth_close(df, in_rth)

        # ── Step 3: Gap (known at the session-open bar) ────────────────────────
        gap = sess_open - prior_close                       # +ve = gap up
        gap_abs = gap.abs()
        gap_threshold = atr * p["gap_min_atr"]
        gap_ok = gap_abs > gap_threshold

        gap_up   = in_rth & gap_ok & (gap > 0) & prior_close.notna()
        gap_down = in_rth & gap_ok & (gap < 0) & prior_close.notna()

        # ── Step 4: Causal session extreme since open (cumulative) ─────────────
        # cummax(high)/cummin(low) WITHIN the current session, up to & including
        # the current bar.  On the open bar this is just that bar's high/low.
        sess_high_so_far, sess_low_so_far = self._session_running_extremes(
            df, sess_id
        )

        # ── Step 5: Entry-bar selection ────────────────────────────────────────
        # Enter on the open bar (bar_in_sess == confirm_bars).  confirm_bars=0
        # means the open bar itself; confirm_bars=N means the (N+1)-th RTH bar,
        # which lets the open's gap be "confirmed" by a few bars of price action.
        # All of these are causal: bar_in_sess is known at each bar.
        entry_bar = in_rth & (bar_in_sess == int(p["confirm_bars"]))

        # gap_up / gap_down were computed using sess_open and prior_close which
        # are constant within the session, so they are valid on the entry bar.
        short_setup = gap_up   & entry_bar      # fade gap up  → short
        long_setup  = gap_down & entry_bar      # fade gap down → long

        # ── Step 6: Confluence weighting ───────────────────────────────────────
        bias = self.htf_bias(df)
        rsi  = df.get("rsi", pd.Series(50.0, index=df.index)).fillna(50.0)
        vol_ok = self.volatility_filter(df)

        # For a fade-short we want bias not strongly bullish / rsi not deeply
        # oversold; for a fade-long the opposite.  These are soft confluence
        # points, not hard gates (the gap itself is the edge).
        use_bias = bool(p["use_bias"])

        long_conf = (
            in_rth.astype(int) +                          # [1] in RTH session
            gap_ok.astype(int) +                          # [2] gap large enough
            entry_bar.astype(int) +                       # [3] on entry bar
            (((bias >= 0) | (~use_bias)).astype(int)) +   # [4] bias not against fade
            (rsi < 70).astype(int) +                      # [5] not already overbought
            vol_ok.astype(int)                            # [6] volatility sane
        )
        short_conf = (
            in_rth.astype(int) +
            gap_ok.astype(int) +
            entry_bar.astype(int) +
            (((bias <= 0) | (~use_bias)).astype(int)) +
            (rsi > 30).astype(int) +
            vol_ok.astype(int)
        )

        long_trigger  = long_setup  & (long_conf  >= self.min_confluence)
        short_trigger = short_setup & (short_conf >= self.min_confluence)

        out["long_signal"]  = long_trigger
        out["short_signal"] = short_trigger

        # ── Step 7: Entry / stop / target ──────────────────────────────────────
        out.loc[long_trigger,  "entry_price"] = df.loc[long_trigger,  "close"]
        out.loc[short_trigger, "entry_price"] = df.loc[short_trigger, "close"]

        atr_sl = atr * p["atr_sl_mult"]

        # Stop sits beyond the causal session extreme since open.
        out.loc[long_trigger, "stop_loss"] = (
            sess_low_so_far.loc[long_trigger] - atr_sl.loc[long_trigger]
        )
        out.loc[short_trigger, "stop_loss"] = (
            sess_high_so_far.loc[short_trigger] + atr_sl.loc[short_trigger]
        )

        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()

        # Primary target = gap-fill (prior RTH close).  Floor it to rr_ratio.
        rr = float(p["rr_ratio"])

        # LONG (gap down → fade up): target is prior_close (above entry).
        long_fill   = prior_close
        long_rr_tgt = out["entry_price"] + sl_dist * rr
        # take the farther (more conservative reward) of the two, but always >= entry
        out.loc[long_trigger, "take_profit"] = np.maximum(
            long_fill.loc[long_trigger].to_numpy(),
            long_rr_tgt.loc[long_trigger].to_numpy(),
        )

        # SHORT (gap up → fade down): target is prior_close (below entry).
        short_fill   = prior_close
        short_rr_tgt = out["entry_price"] - sl_dist * rr
        out.loc[short_trigger, "take_profit"] = np.minimum(
            short_fill.loc[short_trigger].to_numpy(),
            short_rr_tgt.loc[short_trigger].to_numpy(),
        )

        out["confluence_count"] = np.where(
            long_trigger,  long_conf,
            np.where(short_trigger, short_conf, 0),
        )

        return self._finalize_signals(out, self.min_confluence)

    # ── Causal session helpers ──────────────────────────────────────────────────

    @staticmethod
    def _session_mask(df: pd.DataFrame, session: str) -> pd.Series:
        """True for bars inside *session* (UTC).  Pointwise on the index → causal."""
        sess = SESSIONS.get(session)
        if sess is None:
            raise ValueError(f"Unknown session '{session}'. Valid: {list(SESSIONS)}")

        def _tf(t: str) -> float:
            h, m = t.split(":")
            return int(h) + int(m) / 60.0

        hour = df.index.hour + df.index.minute / 60.0
        start = _tf(sess["start"])
        end   = _tf(sess["end"])
        return pd.Series((hour >= start) & (hour < end), index=df.index)

    @staticmethod
    def _session_open_and_index(df: pd.DataFrame, in_rth: pd.Series):
        """
        For every bar return:
          sess_open    — the OPEN price of the first RTH bar of the current
                         session (constant within a session, NaN outside RTH).
          sess_id      — an integer id, incremented on each transition INTO an
                         RTH session; NaN outside RTH.
          bar_in_sess  — 0-based position of the bar within its RTH session
                         (0 = the open bar); -1 outside RTH.

        Everything is built from a left-to-right scan: a value at bar t depends
        only on bars <= t.  No look-ahead.
        """
        rth = in_rth.to_numpy()
        opens = df["open"].to_numpy()
        n = len(df)

        sess_open   = np.full(n, np.nan)
        sess_id_arr = np.full(n, np.nan)
        bar_idx     = np.full(n, -1, dtype=np.int64)

        cur_id = -1
        cur_open = np.nan
        cur_pos = -1
        prev_in = False
        for i in range(n):
            if rth[i]:
                if not prev_in:
                    # transition into a new RTH session
                    cur_id += 1
                    cur_open = opens[i]
                    cur_pos = 0
                else:
                    cur_pos += 1
                sess_open[i]   = cur_open
                sess_id_arr[i] = cur_id
                bar_idx[i]     = cur_pos
            prev_in = rth[i]

        return (
            pd.Series(sess_open, index=df.index),
            pd.Series(sess_id_arr, index=df.index),
            pd.Series(bar_idx, index=df.index),
        )

    @staticmethod
    def _prior_rth_close(df: pd.DataFrame, in_rth: pd.Series) -> pd.Series:
        """
        Prior RTH session close, available from the moment a new RTH session
        opens.  Causal construction:

          1. Mark the LAST bar of each RTH session (last True before a False).
          2. Record that bar's close.
          3. Forward-fill that recorded close.
          4. Because step 3 makes the value available only from the session's
             own last bar onward, by the time the NEXT session opens the carried
             value is the previous session's final close.  The current
             session's own (still-forming) close never leaks back to its open.

        Outside any prior data the value is NaN (filtered out in entry logic).
        """
        rth = in_rth.to_numpy()
        close = df["close"].to_numpy()
        n = len(df)

        last_close = np.full(n, np.nan)
        for i in range(n):
            is_last_rth = rth[i] and (i + 1 >= n or not rth[i + 1])
            if is_last_rth:
                last_close[i] = close[i]

        # forward-fill the captured session-final closes
        prior = pd.Series(last_close, index=df.index).ffill()

        # shift by one bar so a session's OWN final-close value is not visible
        # during that same session; the next session sees it as "prior".
        # (Within a session, prior should be the PREVIOUS session's close.)
        return prior.shift(1)

    @staticmethod
    def _session_running_extremes(df: pd.DataFrame, sess_id: pd.Series):
        """
        Cumulative high / low WITHIN the current RTH session, up to and
        including the current bar.  Implemented as a grouped cummax/cummin so a
        bar at position k only sees bars 0..k of its own session.  Outside RTH
        (sess_id NaN) the values are NaN.
        """
        ids = sess_id.to_numpy()
        high = df["high"].to_numpy()
        low  = df["low"].to_numpy()
        n = len(df)

        run_high = np.full(n, np.nan)
        run_low  = np.full(n, np.nan)

        cur_id = np.nan
        cmax = -np.inf
        cmin = np.inf
        for i in range(n):
            if np.isnan(ids[i]):
                continue
            if ids[i] != cur_id:
                cur_id = ids[i]
                cmax = high[i]
                cmin = low[i]
            else:
                cmax = high[i] if high[i] > cmax else cmax
                cmin = low[i]  if low[i]  < cmin else cmin
            run_high[i] = cmax
            run_low[i]  = cmin

        return (
            pd.Series(run_high, index=df.index),
            pd.Series(run_low, index=df.index),
        )

    @staticmethod
    def _rolling_any(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=1).max().astype(bool)
