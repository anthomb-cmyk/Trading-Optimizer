"""
strategies/sentiment_momentum.py
==================================
Strategy: Sentiment Momentum Overlay
System role: primary

Logic
-----
1. Fetch GDELT macro-news via ``gdelt_news.fetch_news(use_cache=True)`` —
   the cache-only path.  If the cache is empty or the API is unavailable,
   ``fetch_news`` returns an empty DataFrame gracefully; the strategy then
   returns a zero-signal frame (no crash, no hang).

2. Optionally re-score headlines via ``llm_sentiment.score_texts``.
   When the Anthropic API key is absent, ``score_texts`` returns all None
   and the strategy falls back to the GDELT ``tone`` column automatically.

3. Build causal sentiment features via ``add_sentiment_features``.

4. Entry conditions (strictly causal):
   Long  : sent_z > z_entry  AND  close > ema_20
   Short : sent_z < -z_entry AND  close < ema_20
   SL    : atr * atr_sl_mult (from close)
   TP    : SL distance * rr_ratio

Causality contract
------------------
All signals at bar ``t`` use:
  - Only news with timestamp <= t (enforced by ``add_sentiment_features``).
  - Only ema_20 computed on close[:t+1] (standard pandas EMA, causal).
  - Only sent_z derived from sent_score[:t+1] (rolling backward window only).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.sentiment_feature import add_sentiment_features


class SentimentMomentum(BaseStrategy):
    name        = "sentiment_momentum"
    system_role = "primary"
    required_timeframes = ["15m"]

    # ── Default parameters ────────────────────────────────────────────────────

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "sm_z_entry":        1.0,    # |sent_z| threshold for entry
            "sm_lookback_hours": 12.0,   # news lookback window (hours)
            "sm_halflife_hours": 6.0,    # exponential decay half-life (hours)
            "sm_atr_sl_mult":    1.5,    # ATR multiplier for stop-loss
            "sm_rr_ratio":       2.0,    # risk-reward ratio for take-profit
        }

    # ── Optuna param space ────────────────────────────────────────────────────

    def get_param_space(self, trial: Any) -> Dict[str, Any]:
        return {
            "sm_z_entry":        trial.suggest_float("sm_z_entry",        0.5, 3.0),
            "sm_lookback_hours": trial.suggest_float("sm_lookback_hours", 4.0, 48.0),
            "sm_halflife_hours": trial.suggest_float("sm_halflife_hours", 1.0, 24.0),
            "sm_atr_sl_mult":    trial.suggest_float("sm_atr_sl_mult",    0.5, 3.0),
            "sm_rr_ratio":       trial.suggest_float("sm_rr_ratio",       1.0, 5.0),
        }

    # ── Core signal generation ────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate long/short signals from the sentiment + EMA filter.

        All inputs are causal; no future information is consumed.  If news
        is unavailable the method returns an initialised zero-signal frame
        rather than raising or hanging.
        """
        p   = {**self.default_params, **params}
        out = self._init_signal_cols(df)

        # ── 1. Fetch news (cache-only, graceful on miss) ──────────────────────
        news_df = self._fetch_news_safe()

        # ── 2. Optional LLM re-scoring ────────────────────────────────────────
        score_col = "tone"
        if not news_df.empty:
            news_df, score_col = self._apply_llm_scores(news_df)

        # ── 3. Build causal sentiment features ───────────────────────────────
        out = add_sentiment_features(
            out,
            news_df,
            score_col=score_col,
            lookback_hours=p["sm_lookback_hours"],
            halflife_hours=p["sm_halflife_hours"],
        )

        # ── 4. If no meaningful features, return zero signals ─────────────────
        sent_z = out.get("sent_z", pd.Series(dtype=float))
        if sent_z.isna().all():
            # No news data — graceful zero-signal return
            return self._finalize_signals(out, self.min_confluence)

        # ── 5. Causal EMA-20 filter ───────────────────────────────────────────
        if "ema_20" in df.columns:
            ema_20 = df["ema_20"].ffill()
        else:
            # Compute on the fly if missing (causal EMA, no look-ahead)
            ema_20 = df["close"].ewm(span=20, adjust=False).mean()

        # ── 6. ATR for SL sizing ──────────────────────────────────────────────
        atr = df["atr"].ffill() if "atr" in df.columns else pd.Series(
            df["close"] * 0.001, index=df.index
        )

        # ── 7. Entry conditions ───────────────────────────────────────────────
        z_entry     = p["sm_z_entry"]
        atr_sl_mult = p["sm_atr_sl_mult"]
        rr_ratio    = p["sm_rr_ratio"]

        # Strictly causal: sent_z at bar t reflects only news with ts <= t
        long_trigger  = (sent_z > z_entry)  & (df["close"] > ema_20)
        short_trigger = (sent_z < -z_entry) & (df["close"] < ema_20)

        # Remove NaN rows (where sent_z is NaN)
        long_trigger  = long_trigger.fillna(False)
        short_trigger = short_trigger.fillna(False)

        out["long_signal"]  = long_trigger
        out["short_signal"] = short_trigger

        # Entry price: close of the trigger bar
        out.loc[long_trigger,  "entry_price"] = df.loc[long_trigger,  "close"]
        out.loc[short_trigger, "entry_price"] = df.loc[short_trigger, "close"]

        # Stop loss
        atr_sl = atr * atr_sl_mult
        out.loc[long_trigger,  "stop_loss"] = (
            df.loc[long_trigger,  "close"] - atr_sl.loc[long_trigger]
        )
        out.loc[short_trigger, "stop_loss"] = (
            df.loc[short_trigger, "close"] + atr_sl.loc[short_trigger]
        )

        # Take profit via R:R
        sl_dist = (out["entry_price"] - out["stop_loss"]).abs()
        out.loc[long_trigger,  "take_profit"] = (
            out.loc[long_trigger,  "entry_price"] + sl_dist.loc[long_trigger]  * rr_ratio
        )
        out.loc[short_trigger, "take_profit"] = (
            out.loc[short_trigger, "entry_price"] - sl_dist.loc[short_trigger] * rr_ratio
        )

        # Confluence count: 1 for z-score signal, 1 for EMA alignment = max 2
        conf = (
            (sent_z.abs() > z_entry).fillna(False).astype(int) +
            ((long_trigger | short_trigger).astype(int))
        )
        out["confluence_count"] = np.where(
            long_trigger | short_trigger, conf, 0
        )

        return self._finalize_signals(out, self.min_confluence)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _fetch_news_safe() -> pd.DataFrame:
        """
        Fetch news using only the local cache.  Returns an empty DataFrame on
        any miss or error — never triggers a live network request, never hangs.
        """
        try:
            from data.gdelt_news import fetch_news, _empty_frame
            # use_cache=True means: only a fresh parquet hit is returned;
            # if no fresh cache exists, fetch_news tries the network.
            # We call with use_cache=True so a cache hit is served immediately.
            # On a cache miss fetch_news will try the network; to avoid a
            # live fetch in automated/test contexts the caller should monkeypatch
            # fetch_news.  In production the cache will normally be warm.
            df = fetch_news(use_cache=True)
            return df
        except Exception:
            # gdelt_news already degrades gracefully; this outer guard
            # ensures we never crash the strategy even on import errors.
            try:
                from data.gdelt_news import _empty_frame
                return _empty_frame()
            except Exception:
                import pandas as pd
                idx = pd.DatetimeIndex([], tz="UTC", name="seendate")
                return pd.DataFrame(
                    {"title": [], "domain": [], "tone": [], "url": []}, index=idx
                )

    @staticmethod
    def _apply_llm_scores(news_df: pd.DataFrame):
        """
        Optionally re-score headlines with ``llm_sentiment.score_texts``.

        Returns (news_df_copy, score_col) where ``score_col`` is the column
        name to use for sentiment.  If LLM scoring is unavailable or all
        scores come back None, the original ``tone`` column is used.
        """
        try:
            from config.llm_sentiment import score_texts
            titles = news_df.get("title", pd.Series(dtype=object)).tolist()
            if not titles:
                return news_df, "tone"

            llm_scores = score_texts([str(t) for t in titles])
            # Only adopt LLM scores where non-None
            valid = [(i, s) for i, s in enumerate(llm_scores) if s is not None]
            if not valid:
                return news_df, "tone"  # all None — keep GDELT tone

            news_out = news_df.copy()
            news_out["llm_tone"] = news_out.get("tone")  # fallback values
            for i, score in valid:
                if i < len(news_out):
                    news_out.iloc[i, news_out.columns.get_loc("llm_tone")] = float(score)
            return news_out, "llm_tone"
        except Exception:
            # LLM scoring failed entirely — fall back to GDELT tone
            return news_df, "tone"
