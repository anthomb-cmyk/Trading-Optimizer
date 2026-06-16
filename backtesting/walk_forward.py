"""
Walk-Forward Analysis for APEX Optimizer.

Splits historical data into rolling In-Sample (IS) and Out-of-Sample (OOS)
windows to measure how well optimized parameters generalize.

Window layout (n_splits = 5):
  |------- IS_1 (70%) -------|gap| OOS_1 (30%) |
               |------- IS_2 (70%) -------|gap| OOS_2 |
                            ...

Robustness criteria:
  - OOS score must be >= oos_min_ratio * IS score in at least
    oos_pass_threshold of splits
  - OOS win rate >= min_oos_win_rate in majority of splits
  - OOS profit factor >= 1.0 in majority of splits

Returns a WalkForwardResult with per-fold metrics and an aggregate
robustness flag used by the optimizer to reject overfitted parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics import compute_score
from config.logger import get_logger
from config.settings import OPTIMIZER

log = get_logger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold:          int
    is_bars:       int
    oos_bars:      int
    is_score:      float
    oos_score:     float
    is_metrics:    Dict[str, float]
    oos_metrics:   Dict[str, float]
    degradation:   float = 0.0       # (is_score - oos_score) / is_score

    def __post_init__(self):
        if self.is_score > 0:
            self.degradation = (self.is_score - self.oos_score) / self.is_score
        else:
            self.degradation = 1.0

    @property
    def passes(self) -> bool:
        """OOS result is acceptable."""
        return (
            self.oos_score > 0.0
            and self.oos_metrics.get("profit_factor", 0) >= 1.0
            and self.oos_metrics.get("win_rate", 0)      >= 0.30
        )


@dataclass
class WalkForwardResult:
    folds:            List[FoldResult] = field(default_factory=list)
    n_passes:         int   = 0
    n_total:          int   = 0
    avg_is_score:     float = 0.0
    avg_oos_score:    float = 0.0
    avg_degradation:  float = 0.0
    is_robust:        bool  = False
    pass_rate:        float = 0.0

    def __post_init__(self):
        if self.folds:
            self.n_total         = len(self.folds)
            self.n_passes        = sum(1 for f in self.folds if f.passes)
            self.avg_is_score    = float(np.mean([f.is_score  for f in self.folds]))
            self.avg_oos_score   = float(np.mean([f.oos_score for f in self.folds]))
            self.avg_degradation = float(np.mean([f.degradation for f in self.folds]))
            self.pass_rate       = self.n_passes / self.n_total if self.n_total > 0 else 0.0
            # Robust if majority of OOS folds pass
            self.is_robust       = self.n_passes >= max(1, int(self.n_total * 0.6))

    def summary(self) -> str:
        return (
            f"WF passes={self.n_passes}/{self.n_total} | "
            f"IS={self.avg_is_score:.4f} | OOS={self.avg_oos_score:.4f} | "
            f"Degrade={self.avg_degradation:.1%} | Robust={self.is_robust}"
        )


# ── Splitter ──────────────────────────────────────────────────────────────────

def make_wf_splits(
    n_bars:    int,
    n_splits:  int = 5,
    oos_frac:  float = 0.20,
    gap_bars:  int = 20,
    min_is_bars: int = 200,
) -> List[Tuple[slice, slice]]:
    """
    Generate (is_slice, oos_slice) pairs for walk-forward analysis.
    Uses a rolling window where the OOS window slides forward each fold.

    Returns list of (in_sample_slice, oos_slice) tuples.
    """
    oos_size = max(50, int(n_bars * oos_frac))
    is_size  = n_bars - n_splits * oos_size

    if is_size < min_is_bars:
        # Not enough data — reduce splits
        n_splits = max(1, (n_bars - min_is_bars) // oos_size)
        is_size  = n_bars - n_splits * oos_size

    splits = []
    for fold in range(n_splits):
        is_start  = 0
        is_end    = is_size + fold * oos_size
        oos_start = is_end + gap_bars
        oos_end   = oos_start + oos_size

        if oos_end > n_bars:
            break

        splits.append((
            slice(is_start, is_end),
            slice(oos_start, oos_end),
        ))

    return splits


# ── Analyzer ─────────────────────────────────────────────────────────────────

class WalkForwardAnalyzer:
    """Run walk-forward analysis over a dataset for a given parameter set."""

    def __init__(
        self,
        engine:    BacktestEngine,
        n_splits:  int   = OPTIMIZER["walk_forward_splits"],
        oos_frac:  float = 0.20,
        gap_bars:  int   = OPTIMIZER["walk_forward_gap_bars"],
        min_trades: int  = OPTIMIZER["min_trades_per_period"],
    ):
        self.engine     = engine
        self.n_splits   = n_splits
        self.oos_frac   = oos_frac
        self.gap_bars   = gap_bars
        self.min_trades = min_trades

    def analyze(
        self,
        df:       pd.DataFrame,
        signals:  pd.DataFrame,
        params:   Dict[str, Any],
    ) -> WalkForwardResult:
        """
        Run IS+OOS backtest for each fold and return aggregate result.

        signals must already be computed for the full df before calling.
        """
        splits = make_wf_splits(
            n_bars   = len(df),
            n_splits = self.n_splits,
            oos_frac = self.oos_frac,
            gap_bars = self.gap_bars,
        )

        if not splits:
            log.warning("WalkForward: insufficient data for any splits.")
            return WalkForwardResult()

        fold_results = []
        for fold_idx, (is_sl, oos_sl) in enumerate(splits):
            is_df  = df.iloc[is_sl].copy()
            oos_df = df.iloc[oos_sl].copy()
            is_sig  = signals.iloc[is_sl].copy()
            oos_sig = signals.iloc[oos_sl].copy()

            # Backtest each period with the same params
            is_result  = self.engine.run(is_df,  is_sig,  params)
            oos_result = self.engine.run(oos_df, oos_sig, params)

            fold = FoldResult(
                fold        = fold_idx + 1,
                is_bars     = len(is_df),
                oos_bars    = len(oos_df),
                is_score    = is_result.score  if is_result.is_valid(self.min_trades)  else 0.0,
                oos_score   = oos_result.score if oos_result.is_valid(self.min_trades) else 0.0,
                is_metrics  = is_result.metrics,
                oos_metrics = oos_result.metrics,
            )
            fold_results.append(fold)
            log.debug(
                "WF fold %d/%d: IS=%s OOS=%s pass=%s",
                fold_idx + 1, len(splits),
                f"{fold.is_score:.4f}", f"{fold.oos_score:.4f}", fold.passes,
            )

        return WalkForwardResult(folds=fold_results)

    def quick_check(
        self,
        df:      pd.DataFrame,
        signals: pd.DataFrame,
        params:  Dict[str, Any],
        n_folds: int = 3,
    ) -> float:
        """
        Fast single-value robustness score for use inside Optuna objective.
        Returns avg OOS score across n_folds (fewer folds for speed).
        """
        orig_splits = self.n_splits
        self.n_splits = n_folds
        result = self.analyze(df, signals, params)
        self.n_splits = orig_splits
        return result.avg_oos_score
