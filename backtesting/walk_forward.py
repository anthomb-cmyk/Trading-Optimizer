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

CAUSAL SIGNAL GENERATION
-------------------------
Signals are generated *per fold* rather than on the full dataset.  For each
fold the helper `generate_fold_signals` slices the dataframe as:

    [fold_start - warmup_bars : fold_end]

calls ``generate_signals_fn(slice, params)``, then **strips the warmup prefix
rows before returning**, so the backtesting engine never scores warmup bars and
OOS signals are computed only from bars inside their own causal window.

The public ``analyze`` / ``quick_check`` API now accepts a
``generate_signals_fn`` callable ``(df, params) -> signals_df`` instead of a
pre-computed ``signals`` frame.  Call sites in ``sunday_optimizer.py`` have
been updated accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics import compute_score
from config.logger import get_logger
from config.settings import OPTIMIZER, WARMUP_BARS

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


# ── Per-fold causal signal generation ────────────────────────────────────────

def generate_fold_signals(
    df:                   pd.DataFrame,
    fold_slice:           slice,
    generate_signals_fn:  Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
    params:               Dict[str, Any],
    warmup_bars:          int = WARMUP_BARS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate signals for a single fold without look-ahead.

    Slices the dataframe as ``[prefix_start : fold_end]`` where
    ``prefix_start = max(0, fold_start - warmup_bars)``.  Signal generation
    therefore sees only bars up to ``fold_end`` and can use the warmup prefix
    for indicator settle-in.

    Returns
    -------
    fold_df : pd.DataFrame
        The raw OHLCV+indicators slice for the fold (warmup rows excluded).
    fold_signals : pd.DataFrame
        Signals corresponding to ``fold_df`` (warmup rows excluded).

    The warmup prefix is included during signal computation but stripped from
    both return values so the backtesting engine scores only real fold rows.
    """
    fold_start = fold_slice.start or 0
    fold_end   = fold_slice.stop  or len(df)

    prefix_start = max(0, fold_start - warmup_bars)
    n_prefix     = fold_start - prefix_start  # how many warmup rows we prepended

    # Slice with causal warmup prefix
    ctx_df  = df.iloc[prefix_start:fold_end].copy()

    # Generate signals on the causal context window
    ctx_sig = generate_signals_fn(ctx_df, params)

    # Strip the warmup prefix from both df and signals
    fold_df      = ctx_df.iloc[n_prefix:].copy()
    fold_signals = ctx_sig.iloc[n_prefix:].copy()

    return fold_df, fold_signals


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
        df:                   pd.DataFrame,
        generate_signals_fn:  Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
        params:               Dict[str, Any],
        warmup_bars:          int = WARMUP_BARS,
    ) -> WalkForwardResult:
        """
        Run IS+OOS backtest for each fold and return aggregate result.

        Signals are generated *per fold* using ``generate_signals_fn`` so that
        OOS signals are computed from a causal window ending at the fold
        boundary.  A warmup prefix (``warmup_bars`` rows taken from the IS
        side) is prepended to each fold so indicators can settle; those rows
        are stripped before scoring.

        Parameters
        ----------
        df:
            Full raw dataframe (OHLCV + indicators).
        generate_signals_fn:
            Callable ``(df_slice, params) -> signals_df``.  Must be the same
            function used during the normal backtest (e.g.
            ``system.generate_signals``).
        params:
            Strategy parameter dict for this trial.
        warmup_bars:
            Number of bars prepended to each fold as indicator warm-up.
            Defaults to ``WARMUP_BARS`` from settings.
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
            # Generate IS signals causally (warmup taken from earlier history)
            is_df, is_sig = generate_fold_signals(
                df, is_sl, generate_signals_fn, params, warmup_bars,
            )
            # Generate OOS signals causally (warmup taken from IS-side bars)
            oos_df, oos_sig = generate_fold_signals(
                df, oos_sl, generate_signals_fn, params, warmup_bars,
            )

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
        df:                   pd.DataFrame,
        generate_signals_fn:  Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
        params:               Dict[str, Any],
        n_folds:              int = 3,
        warmup_bars:          int = WARMUP_BARS,
    ) -> float:
        """
        Fast single-value robustness score for use inside Optuna objective.
        Returns avg OOS score across n_folds (fewer folds for speed).

        ``generate_signals_fn`` replaces the old pre-computed ``signals``
        argument — signals are now generated per fold to avoid look-ahead.
        """
        orig_splits = self.n_splits
        self.n_splits = n_folds
        result = self.analyze(df, generate_signals_fn, params, warmup_bars)
        self.n_splits = orig_splits
        return result.avg_oos_score
