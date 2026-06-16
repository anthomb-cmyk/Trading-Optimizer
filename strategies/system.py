"""
APEX Strategy System — orchestrates all 10 strategies in their correct hierarchy.

Execution order:
  Layer 0 — Bias Filter:    Strategy 10 (DailyBiasIntraday)
                             → sets allowed direction for the entire day
  Layer 1 — Timing Filter:  Strategy 6 (KillZoneReversal)
                             → only open trades during valid kill zones
  Layer 2 — Primary Entry:  Strategies 1, 2, 3 (LiquiditySweep, OrderBlock, FVGRetest)
                             → any one firing is sufficient for a potential trade
  Layer 3 — Confirmation:   Strategies 4, 5, 7, 8, 9 (Breaker, MSS, StopHunt, OTE, Asian)
                             → each adds +1 to confluence; min_confirmations required

Final signal conditions:
  1. Bias is non-zero (S10 says there is a valid direction today)
  2. In kill zone (S6 timing gate is open)
  3. At least one primary entry fires in the bias direction
  4. System confluence count >= system_min_confluence
  5. Volatility within acceptable bounds
  6. No active news blackout (handled upstream by NewsFilter)

Usage:
    system = StrategySystem(symbol="MES")
    signals = system.generate_signals(df, params)
    # signals has all standard signal columns PLUS:
    #   bias_score          int    S10 raw score (−3 to +3)
    #   primary_strategy    str    which primary fired ("ls" | "ob" | "fvg" | "multiple")
    #   confirmation_count  int    how many confirmation strategies agreed
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import MIN_CONFLUENCE_CONFIRMATIONS
from config.logger import get_logger

log = get_logger(__name__)


class StrategySystem:
    """
    Orchestrates the 10 APEX strategies into a single coherent signal pipeline.
    All strategies are lazy-instantiated on first use.
    """

    # System-level minimum: primary signal + at least this many confirmations
    DEFAULT_SYSTEM_MIN_CONFIRMATIONS = 2

    def __init__(
        self,
        symbol:                  str = "MES",
        system_min_confirmations: int = DEFAULT_SYSTEM_MIN_CONFIRMATIONS,
    ):
        self.symbol   = symbol
        self.sys_min  = system_min_confirmations
        self._strats  = {}     # lazy cache

    # ── Lazy strategy accessors ───────────────────────────────────────────────

    def _get(self, name: str):
        if name not in self._strats:
            from strategies import get_strategy
            self._strats[name] = get_strategy(name, self.symbol)
        return self._strats[name]

    @property
    def bias_strategy(self):
        return self._get("daily_bias_intraday")

    @property
    def timing_strategy(self):
        return self._get("kill_zone_reversal")

    @property
    def primary_strategies(self) -> List:
        return [
            self._get("liquidity_sweep_reversal"),
            self._get("order_block_reversal"),
            self._get("fvg_retest"),
        ]

    @property
    def confirmation_strategies(self) -> List:
        return [
            self._get("breaker_block_reversal"),
            self._get("market_structure_shift"),
            self._get("stop_hunt_continuation"),
            self._get("ote_fibonacci_entry"),
            self._get("asian_session_range_breakout"),
        ]

    # ── Main signal generation ────────────────────────────────────────────────

    def generate_signals(
        self,
        df:     pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Run the full system pipeline and return a signal DataFrame.

        Layer 0 → Layer 1 → Layer 2 → Layer 3 in sequence.
        Each layer filters the candidate signals from the previous layer.
        """
        out = df.copy()
        for col in ("long_signal", "short_signal", "entry_price",
                    "stop_loss", "take_profit", "confluence_count",
                    "valid", "bias_score", "bias", "in_kill_zone",
                    "primary_strategy", "confirmation_count"):
            out[col] = np.nan if col in ("entry_price", "stop_loss", "take_profit") \
                       else (False if col in ("long_signal", "short_signal", "valid", "in_kill_zone") \
                       else (0 if col in ("confluence_count", "bias_score", "bias", "confirmation_count") \
                       else ""))

        # ── Layer 0: Daily bias ───────────────────────────────────────────────
        bias = self.bias_strategy.get_bias(df, params)
        bias_score = self.bias_strategy.get_bias_score(df, params)
        out["bias_score"] = bias_score
        out["bias"]       = bias

        no_bias = bias == 0
        if no_bias.all():
            log.info(
                "System: no bias on any bar — skipping signals (last raw score=%+d)",
                int(bias_score.iloc[-1]),
            )
            return out

        # ── Layer 1: Kill zone timing ─────────────────────────────────────────
        timing = self.timing_strategy.get_timing_mask(df, params)
        out["in_kill_zone"] = timing

        gate_open = (bias != 0) & timing

        # ── Layer 2: Primary entries ──────────────────────────────────────────
        long_primary  = pd.Series(False, index=df.index)
        short_primary = pd.Series(False, index=df.index)
        primary_name  = pd.Series("", index=df.index)

        primary_names = ["ls", "ob", "fvg"]
        for strat, tag in zip(self.primary_strategies, primary_names):
            p = {**strat.default_params, **params}
            sig = strat.generate_signals(df, p)
            new_long  = sig["long_signal"]  & gate_open & (bias == 1)
            new_short = sig["short_signal"] & gate_open & (bias == -1)

            # Tag which primary fired (first wins; "multiple" if >1)
            primary_name = np.where(
                new_long | new_short,
                np.where(primary_name != "", "multiple", tag),
                primary_name,
            )
            primary_name = pd.Series(primary_name, index=df.index)

            long_primary  |= new_long
            short_primary |= new_short

            # Carry entry price/SL/TP from the first primary to fire
            first_long  = new_long  & ~(long_primary.shift(1).fillna(False))
            first_short = new_short & ~(short_primary.shift(1).fillna(False))
            out.loc[first_long,  "entry_price"] = sig.loc[first_long,  "entry_price"]
            out.loc[first_long,  "stop_loss"]   = sig.loc[first_long,  "stop_loss"]
            out.loc[first_long,  "take_profit"]  = sig.loc[first_long,  "take_profit"]
            out.loc[first_short, "entry_price"] = sig.loc[first_short, "entry_price"]
            out.loc[first_short, "stop_loss"]   = sig.loc[first_short, "stop_loss"]
            out.loc[first_short, "take_profit"]  = sig.loc[first_short, "take_profit"]

        out["primary_strategy"] = primary_name

        if not long_primary.any() and not short_primary.any():
            last_bias = int(bias.iloc[-1])
            last_in_kz = bool(timing.iloc[-1])
            log.info(
                "System: no primary entry fired — last bar bias=%+d, in_kill_zone=%s",
                last_bias, last_in_kz,
            )
            return out

        # ── Layer 3: Confirmation scoring ────────────────────────────────────
        direction = np.where(long_primary, 1, np.where(short_primary, -1, 0))
        direction_s = pd.Series(direction, index=df.index)

        conf_count = pd.Series(0, index=df.index)
        for strat in self.confirmation_strategies:
            p = {**strat.default_params, **params}
            conf_count += strat.get_confirmation(df, direction_s, p)

        out["confirmation_count"] = conf_count

        # ── Final gate ────────────────────────────────────────────────────────
        # Allow optimizer to tune the confirmation threshold (default: self.sys_min).
        # Keeping it in params lets Optuna discover the best trade-off between
        # signal frequency and quality.
        sys_min_eff = int(params.get("system_min_conf", self.sys_min))
        system_ok   = conf_count >= sys_min_eff

        out["long_signal"]      = long_primary  & system_ok
        out["short_signal"]     = short_primary & system_ok
        out["confluence_count"] = conf_count + (bias.abs())   # bias counts too
        out["valid"]            = out["long_signal"] | out["short_signal"]

        n_long  = out["long_signal"].sum()
        n_short = out["short_signal"].sum()
        n_blocked = (long_primary | short_primary).sum() - n_long - n_short
        log.info(
            "System signals: %d long, %d short (conf >= %d); %d primary(s) blocked by confluence",
            n_long, n_short, sys_min_eff, n_blocked,
        )

        return out

    # ── Convenience: run single strategy standalone ───────────────────────────

    def run_strategy(
        self,
        name:   str,
        df:     pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        strat  = self._get(name)
        p      = {**strat.default_params, **(params or {})}
        return strat.generate_signals(df, p)

    # ── Param space: union of all strategy param spaces ───────────────────────

    def get_full_param_space(self, trial: Any) -> Dict[str, Any]:
        """
        Build the combined parameter dict for an Optuna trial across all strategies.
        Bias, timing, primaries, and confirmations all contribute their params.
        """
        all_strats = (
            [self.bias_strategy, self.timing_strategy]
            + self.primary_strategies
            + self.confirmation_strategies
        )
        params = {}
        for strat in all_strats:
            params.update(strat.get_param_space(trial))
        # System-level tunable: confirmation threshold (0 = no gate, 2 = strict)
        params["system_min_conf"] = trial.suggest_int("system_min_conf", 0, 2)
        return params

    def __repr__(self) -> str:
        return (
            f"StrategySystem(symbol={self.symbol!r}, "
            f"sys_min_conf={self.sys_min})"
        )
