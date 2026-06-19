"""
strategies/detectors.py
=======================
Pure causal SMC detector functions — numpy + pandas only, no heavy deps.

All five functions were originally static methods on BaseStrategy.  They are
extracted here so that:
  1. tests/test_detector_causality.py can import them without pulling in
     config.settings (which requires dotenv / env vars).
  2. The causality guarantee is testable in isolation.

BaseStrategy re-imports every function from this module so all 10 strategies
continue to work without any changes.

Causality contract
------------------
At decision bar ``t``, a detector's output at index ``t`` must depend ONLY
on data at indices <= t.  Equivalently: appending future bars must not alter
any already-emitted value (append-invariance / no-repaint).

Detector verdicts
-----------------
detect_fvg          — CAUSAL AS-IS.  shift(2) = past only; flag at bar i
                      uses i-2 and i (both ≤ i).

detect_fvg_zones    — CAUSAL AS-IS.  Same reasoning as detect_fvg.

detect_order_blocks — FIXED (was look-ahead).
                      Old code: strong_bull.shift(-j) marks OB candle i
                      at time i before displacement at i+j is known.
                      Fix: the OB's existence is flagged at the displacement
                      bar (i+j), the first bar at which all evidence is
                      available.  Retests fire even later, so trading
                      semantics are fully preserved.

detect_bos          — CAUSAL AS-IS.  rolling().max().shift(+N) lags data,
                      never leads.

detect_choch        — CAUSAL AS-IS.  shift(1) on past swing prices, negative
                      shifts only.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ── FVG ───────────────────────────────────────────────────────────────────────

def detect_fvg(
    df: pd.DataFrame,
    min_atr_mult: float = 0.3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Fair Value Gap detection (3-candle imbalance).

    Bullish FVG: candle[i-2].high < candle[i].low  — unfilled gap above
    Bearish FVG: candle[i-2].low  > candle[i].high — unfilled gap below

    Causality: flag at index i uses i-2 (shift+2) and i — fully causal.

    Returns (bull_fvg, bear_fvg) boolean Series marking the MIDDLE candle.
    """
    atr     = df["atr"].ffill()
    min_gap = atr * min_atr_mult

    high_2 = df["high"].shift(2)
    low_2  = df["low"].shift(2)

    bull_fvg = (high_2 < df["low"])  & ((df["low"]  - high_2) >= min_gap)
    bear_fvg = (low_2  > df["high"]) & ((low_2 - df["high"])  >= min_gap)

    return bull_fvg.fillna(False), bear_fvg.fillna(False)


# ── FVG zones ─────────────────────────────────────────────────────────────────

def detect_fvg_zones(
    df: pd.DataFrame,
    min_atr_mult: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns DataFrames for open (unfilled) FVG zones.
    Each row: top, bottom, midpoint of the gap, age (bars since formation).

    Bullish FVG zone: bottom = candle[i-2].high, top = candle[i].low
    Bearish FVG zone: top = candle[i-2].low, bottom = candle[i].high

    Causality: same as detect_fvg — shift(2) is past-only.
    """
    atr     = df["atr"].ffill()
    min_gap = atr * min_atr_mult
    high_2  = df["high"].shift(2)
    low_2   = df["low"].shift(2)

    bull_mask = (high_2 < df["low"])  & ((df["low"]  - high_2) >= min_gap)
    bear_mask = (low_2  > df["high"]) & ((low_2 - df["high"])  >= min_gap)

    bull_zones = pd.DataFrame(index=df.index)
    bull_zones["top"]    = df["low"].where(bull_mask)
    bull_zones["bottom"] = high_2.where(bull_mask)
    bull_zones["mid"]    = (bull_zones["top"] + bull_zones["bottom"]) / 2

    bear_zones = pd.DataFrame(index=df.index)
    bear_zones["top"]    = low_2.where(bear_mask)
    bear_zones["bottom"] = df["high"].where(bear_mask)
    bear_zones["mid"]    = (bear_zones["top"] + bear_zones["bottom"]) / 2

    return bull_zones, bear_zones


# ── Order Blocks ──────────────────────────────────────────────────────────────

def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 20,
    body_pct: float = 0.60,
    displacement_atr_mult: float = 0.8,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Order Block detection via displacement confirmation.

    Bullish OB: last bearish candle immediately before a bullish impulse
                (the impulse bar has strong body AND range >= displacement_atr_mult * ATR)
    Bearish OB: last bullish candle immediately before a bearish impulse

    CAUSALITY FIX (T1.2)
    --------------------
    Old code used ``strong_bull.shift(-j)`` (negative = forward shift) to look
    j bars into the future from the OB candle.  This made the OB known at bar
    i before the displacement at bar i+j had occurred — look-ahead.

    Fix: the OB zone's existence is flagged at bar i+j (the displacement bar),
    the first moment all evidence is available from past data only.

    Concretely: for each bar i that is a potential OB candle (bearish for
    bull_ob, bullish for bear_ob), we scan forward j = 1..lookback for the
    first displacement candle.  We then mark bull_ob / bear_ob TRUE at bar i+j
    (the displacement bar), not at bar i.

    This delays visibility by at most ``lookback`` bars relative to the old
    (leaking) code, but entries only fire on a RETEST of the zone, which
    happens even later — so the trading semantics (buy/sell on retest) are
    fully preserved.

    Returns a 6-tuple of Series indexed like df:
        (bull_ob, bear_ob, bull_ob_high, bull_ob_low, bear_ob_high, bear_ob_low)

    bull_ob / bear_ob
        Boolean Series.  The True flag sits at the DISPLACEMENT bar (the first
        bar at which the OB zone is confirmed from causal data).

    bull_ob_high / bull_ob_low / bear_ob_high / bear_ob_low
        Float Series carrying the OB CANDLE's (bar i) high/low to the
        displacement (confirmation) bar i+j.  NaN everywhere else.
        Consumers should ``.ffill()`` them to propagate the zone forward.
        Causality is preserved: only bar-i data is written, at index i+j >= i.
    """
    atr        = df["atr"].ffill()
    body_ratio = df["body"] / df["range"].replace(0, np.nan)

    # Displacement candles: strong directional body + large range
    strong_bull = (
        df["bullish"] &
        (body_ratio >= body_pct) &
        (df["range"] >= atr * displacement_atr_mult)
    )
    strong_bear = (
        ~df["bullish"] &
        (body_ratio >= body_pct) &
        (df["range"] >= atr * displacement_atr_mult)
    )

    n = len(df)
    bull_ob_arr = np.zeros(n, dtype=bool)
    bear_ob_arr = np.zeros(n, dtype=bool)

    # Zone price arrays: carry the OB candle's high/low to the flag bar
    bull_ob_high_arr = np.full(n, np.nan)
    bull_ob_low_arr  = np.full(n, np.nan)
    bear_ob_high_arr = np.full(n, np.nan)
    bear_ob_low_arr  = np.full(n, np.nan)

    bullish_arr     = df["bullish"].values
    strong_bull_arr = strong_bull.values
    strong_bear_arr = strong_bear.values
    high_arr        = df["high"].values
    low_arr         = df["low"].values

    # Causal scan: iterate each bar i as a potential OB candle.
    # The OB flag lands at bar i+j (the displacement bar), the first
    # bar from which the OB's existence is knowable without look-ahead.
    for i in range(n - 1):
        # Potential bearish OB candle (precursor to a bullish impulse)
        if not bullish_arr[i]:
            for j in range(1, min(lookback + 1, n - i)):
                if strong_bull_arr[i + j]:
                    # Flag the OB at displacement bar i+j (causal)
                    bull_ob_arr[i + j]      = True
                    # Carry the OB candle's zone to the displacement bar
                    bull_ob_high_arr[i + j] = high_arr[i]
                    bull_ob_low_arr[i + j]  = low_arr[i]
                    break  # first displacement wins; stop scanning

        # Potential bullish OB candle (precursor to a bearish impulse)
        if bullish_arr[i]:
            for j in range(1, min(lookback + 1, n - i)):
                if strong_bear_arr[i + j]:
                    # Flag the OB at displacement bar i+j (causal)
                    bear_ob_arr[i + j]      = True
                    # Carry the OB candle's zone to the displacement bar
                    bear_ob_high_arr[i + j] = high_arr[i]
                    bear_ob_low_arr[i + j]  = low_arr[i]
                    break  # first displacement wins; stop scanning

    bull_ob      = pd.Series(bull_ob_arr,      index=df.index)
    bear_ob      = pd.Series(bear_ob_arr,      index=df.index)
    bull_ob_high = pd.Series(bull_ob_high_arr, index=df.index)
    bull_ob_low  = pd.Series(bull_ob_low_arr,  index=df.index)
    bear_ob_high = pd.Series(bear_ob_high_arr, index=df.index)
    bear_ob_low  = pd.Series(bear_ob_low_arr,  index=df.index)
    return bull_ob, bear_ob, bull_ob_high, bull_ob_low, bear_ob_high, bear_ob_low


# ── BOS ───────────────────────────────────────────────────────────────────────

def detect_bos(
    df: pd.DataFrame,
    lookback: int = 20,
    confirm_bars: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """
    Break of Structure.
    Bullish BOS: close above the highest high of the previous lookback bars.
    Bearish BOS: close below the lowest low.

    Causality: rolling().max().shift(+confirm_bars) shifts BACKWARD (lags),
    never forward.  Causal as-is.
    """
    rolling_high = df["high"].rolling(lookback).max().shift(confirm_bars)
    rolling_low  = df["low"].rolling(lookback).min().shift(confirm_bars)
    bull_bos = (df["close"] > rolling_high).fillna(False)
    bear_bos = (df["close"] < rolling_low).fillna(False)
    return bull_bos, bear_bos


# ── CHoCH ─────────────────────────────────────────────────────────────────────

def detect_choch(
    df: pd.DataFrame,
    lookback: int = 20,          # kept for API compatibility; unused internally
) -> Tuple[pd.Series, pd.Series]:
    """
    Change of Character: first close through the most recent internal swing.

    Bullish CHoCH: in a downtrend, closes above the last swing high.
    Bearish CHoCH: in an uptrend, closes below the last swing low.

    Causality: swing_high_price / swing_low_price are forward-filled from
    already-confirmed swing pivots (see data/swings.py which places flags at
    the confirmation bar i+lb).  The additional shift(1) here ensures we
    compare today's close against yesterday's last-known swing — causal.
    """
    last_sh = df["swing_high_price"].ffill()
    last_sl = df["swing_low_price"].ffill()

    bull_choch = (
        (df["close"] > last_sh.shift(1)) &
        (df["close"].shift(1) <= last_sh.shift(1))
    ).fillna(False)
    bear_choch = (
        (df["close"] < last_sl.shift(1)) &
        (df["close"].shift(1) >= last_sl.shift(1))
    ).fillna(False)

    return bull_choch, bear_choch
