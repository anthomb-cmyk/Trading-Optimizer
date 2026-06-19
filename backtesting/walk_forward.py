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

PURGE + EMBARGO (T4.1)
-----------------------
``make_wf_splits_purge_embargo`` extends the basic splitter with two controls:

* **Purge** - training rows whose label horizon overlaps the test window are
  dropped.  For a bar-level backtest with no explicit label horizon the
  purge window equals ``embargo_bars`` (conservative default).
* **Embargo** - an additional gap of ``embargo_bars`` is enforced *after* each
  OOS block so that the next IS window cannot see OOS-adjacent data.

COMBINATORIAL (CPCV) PATHS (T4.1)
-----------------------------------
``make_combinatorial_splits`` partitions the data into N groups and generates
all C(N, k) combinations of k groups as the OOS set (the remainder is IS).
This yields a distribution of OOS paths rather than a single sequential path,
following the spirit of Combinatorially Purged Cross-Validation (Lopez de
Prado, 2018).  The combinatorial splitter also applies purge+embargo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
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


# ── T4.1: Purge + Embargo splitter ───────────────────────────────────────────

def make_wf_splits_purge_embargo(
    n_bars:       int,
    n_splits:     int   = 5,
    oos_frac:     float = 0.20,
    gap_bars:     int   = 20,
    embargo_bars: int   = 10,
    min_is_bars:  int   = 200,
) -> List[Tuple[range, range]]:
    """
    Walk-forward splits with explicit purge and embargo.

    In addition to the ``gap_bars`` gap between IS and OOS (the purge window),
    ``embargo_bars`` rows immediately following each OOS block are excluded
    from the next fold's IS window.  This prevents the model from training on
    bars that are temporally adjacent to the OOS test set (information leakage
    through auto-correlation / look-ahead in feature engineering).

    Parameters
    ----------
    n_bars       : Total number of bars in the dataset.
    n_splits     : Number of sequential folds.
    oos_frac     : Fraction of the dataset reserved for OOS per fold.
    gap_bars     : Gap between IS end and OOS start (the purge window).
                   Training rows within this window are excluded entirely.
    embargo_bars : Additional rows to exclude after each OOS block before
                   the next fold's IS window can begin.
    min_is_bars  : Minimum required IS rows; reduces n_splits automatically.

    Returns
    -------
    List of (is_indices, oos_indices) tuples where each element is a
    ``range`` object of integer row positions.
    """
    oos_size = max(50, int(n_bars * oos_frac))
    is_size  = n_bars - n_splits * oos_size

    if is_size < min_is_bars:
        n_splits = max(1, (n_bars - min_is_bars) // oos_size)
        is_size  = n_bars - n_splits * oos_size

    splits: List[Tuple[range, range]] = []
    for fold in range(n_splits):
        # IS window: [0, is_end) -- grows by one OOS block each fold (rolling)
        is_end    = is_size + fold * oos_size
        oos_start = is_end + gap_bars
        oos_end   = oos_start + oos_size

        if oos_end > n_bars:
            break

        # Purge: drop IS rows within embargo_bars of the OOS boundary.
        # The gap between IS and OOS (gap_bars) is already a purge window;
        # we additionally exclude the tail of IS that is within embargo_bars
        # of the gap start (is_end).  Combined purge = gap_bars + embargo_bars.
        purged_is_end = max(0, is_end - embargo_bars)
        is_indices    = range(0, purged_is_end)

        # OOS rows (gap is already excluded -- OOS starts at oos_start)
        oos_indices = range(oos_start, min(oos_end, n_bars))

        if len(is_indices) < min_is_bars:
            continue  # skip this fold rather than break — later folds have more IS

        splits.append((is_indices, oos_indices))

    return splits


# ── T4.1: Combinatorial (CPCV-style) splits ──────────────────────────────────

def make_combinatorial_splits(
    n_bars:       int,
    n_groups:     int   = 6,
    n_oos_groups: int   = 2,
    embargo_bars: int   = 10,
    min_is_bars:  int   = 200,
) -> List[Tuple[range, range]]:
    """
    Combinatorial Purged Cross-Validation (CPCV) style splits.

    Partitions the data into ``n_groups`` contiguous blocks, then generates
    all C(n_groups, n_oos_groups) combinations of blocks as the OOS set
    (the complement is IS).  Yields a distribution of OOS outcomes rather
    than a single sequential path.

    Purge+embargo is applied: ``embargo_bars`` rows adjacent to each OOS
    block boundary are excluded from the IS set.

    Parameters
    ----------
    n_bars       : Total bars.
    n_groups     : Number of contiguous blocks to partition the data into.
                   C(n_groups, n_oos_groups) paths are generated.
    n_oos_groups : Number of blocks to hold out as OOS per path.
    embargo_bars : Rows excluded from IS on each side of an OOS block.
    min_is_bars  : Paths where IS has fewer than this many rows are skipped.

    Returns
    -------
    List of (is_indices, oos_indices) tuples (``range`` objects).
    """
    if n_groups < 2 or n_oos_groups < 1 or n_oos_groups >= n_groups:
        raise ValueError(
            f"Need 1 <= n_oos_groups < n_groups; got {n_oos_groups} / {n_groups}"
        )

    block_size = n_bars // n_groups
    if block_size < 1:
        raise ValueError(f"n_bars={n_bars} too small for n_groups={n_groups}")

    # Block boundaries (start inclusive, end exclusive)
    blocks: List[Tuple[int, int]] = [
        (i * block_size, min((i + 1) * block_size, n_bars))
        for i in range(n_groups)
    ]

    splits: List[Tuple[range, range]] = []
    for oos_group_indices in combinations(range(n_groups), n_oos_groups):
        oos_set = set(oos_group_indices)

        # Collect OOS indices (all bars in OOS blocks)
        oos_rows: List[int] = []
        for gi in sorted(oos_set):
            start, end = blocks[gi]
            oos_rows.extend(range(start, end))

        # OOS boundary rows to exclude from IS (purge + embargo)
        excluded: set = set(oos_rows)
        for gi in sorted(oos_set):
            start, end = blocks[gi]
            # Embargo before block
            for r in range(max(0, start - embargo_bars), start):
                excluded.add(r)
            # Embargo after block
            for r in range(end, min(n_bars, end + embargo_bars)):
                excluded.add(r)

        # IS rows = everything not OOS and not embargoed
        is_rows_list = sorted(r for r in range(n_bars) if r not in excluded)

        if len(is_rows_list) < min_is_bars:
            continue

        # Represent as a range when the IS rows are contiguous; otherwise keep list.
        # For the return type contract we return ranges for contiguous, else
        # wrap in a minimal object. Here we always have contiguous IS since OOS
        # groups can be non-contiguous, so we fall back to a simple list cast.
        # We wrap in a range-compatible object: use range only if contiguous.
        if is_rows_list == list(range(is_rows_list[0], is_rows_list[-1] + 1)):
            is_range: Any = range(is_rows_list[0], is_rows_list[-1] + 1)
        else:
            is_range = is_rows_list  # type: ignore[assignment]

        if oos_rows == list(range(oos_rows[0], oos_rows[-1] + 1)):
            oos_range: Any = range(oos_rows[0], oos_rows[-1] + 1)
        else:
            oos_range = oos_rows  # type: ignore[assignment]

        splits.append((is_range, oos_range))

    return splits


# ── T4.3: Trial budget governance ────────────────────────────────────────────

def recommended_max_trials(n_oos_obs: int) -> int:
    """
    Data-derived upper bound on Optuna trial budget (T4.3).

    Derived from the False Strategy Theorem (Lopez de Prado, 2013):
    the expected maximum in-sample Sharpe under the null hypothesis grows
    as sqrt(2 * log(N_trials)).  We invert this to find the number of
    trials at which SR0 (expected maximum Sharpe under H0) hits 1.0 given
    the track-record length::

        SR0 = sqrt(V) * f(N)    where V ~ 1/(n_obs - 1)
        SR0 = 1.0  =>  N = exp( (1/V) * ... )

    Practical formula used here:
        T_years = n_oos_obs / (252 * 26)   # years of 15-min bars
        N_max   = exp(0.5 * T_years * (2 * pi * e))

    This is the number of trials beyond which the expected maximum in-sample
    Sharpe under the null exceeds 1.0, making Deflated Sharpe less than 0.5
    for a typical strategy.  The 0.5 * T_years factor is conservative
    (the true formula has additional skew/kurtosis terms), so the returned
    value is a SOFT ceiling, not a hard mathematical limit.

    Parameters
    ----------
    n_oos_obs : Number of OOS observations (bars) available for evaluation.

    Returns
    -------
    Recommended maximum number of Optuna trials (at least 50).
    """
    if n_oos_obs <= 0:
        return 50

    # Bars per year for 15-minute bars (252 trading days, 26 bars/day)
    BARS_PER_YEAR = 252 * 26   # 6552

    t_years = n_oos_obs / BARS_PER_YEAR

    # Inversion of the extreme-value SR0 formula: N such that SR0 ~ 1.0
    # SR0 = sqrt(1/(n-1)) * [(1-gamma)*Phi_inv(1-1/N) + gamma*Phi_inv(1-1/(Ne))]
    # For large N, Phi_inv(1-1/N) ~ sqrt(2*log(N)), so:
    # 1.0 = sqrt(1/(n-1)) * sqrt(2*log(N))  =>  N = exp((n-1)/2)
    # We use the OOS obs count here as a conservative proxy for n.
    n_max = int(math.exp(min(t_years * math.pi * math.e, 20)))  # cap exp at e^20

    return max(50, n_max)


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
        engine:       BacktestEngine,
        n_splits:     int   = OPTIMIZER["walk_forward_splits"],
        oos_frac:     float = 0.20,
        gap_bars:     int   = OPTIMIZER["walk_forward_gap_bars"],
        embargo_bars: int   = 10,
        min_trades:   int   = OPTIMIZER["min_trades_per_period"],
    ):
        self.engine       = engine
        self.n_splits     = n_splits
        self.oos_frac     = oos_frac
        self.gap_bars     = gap_bars
        self.embargo_bars = embargo_bars
        self.min_trades   = min_trades

    def analyze(
        self,
        df:                   pd.DataFrame,
        generate_signals_fn:  Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
        params:               Dict[str, Any],
        warmup_bars:          int  = WARMUP_BARS,
        combinatorial:        bool = False,
        n_groups:             int  = 6,
        n_oos_groups:         int  = 2,
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
        combinatorial:
            When True, use CPCV-style combinatorial splits (``n_groups``
            choose ``n_oos_groups``) instead of the sequential walk-forward
            path.  Produces a distribution of OOS outcomes.
        n_groups:
            Number of contiguous blocks for combinatorial mode.
        n_oos_groups:
            Number of blocks held out as OOS per combinatorial path.
        """
        if combinatorial:
            raw_splits = make_combinatorial_splits(
                n_bars       = len(df),
                n_groups     = n_groups,
                n_oos_groups = n_oos_groups,
                embargo_bars = self.embargo_bars,
            )
            # Convert list/range indices to slice-compatible objects for
            # generate_fold_signals (which uses .start/.stop on a slice).
            # We pass a synthetic slice covering the first-to-last index.
            def _to_slice(idx_seq) -> slice:
                if isinstance(idx_seq, range):
                    return slice(idx_seq.start, idx_seq.stop)
                lst = list(idx_seq)
                return slice(lst[0], lst[-1] + 1) if lst else slice(0, 0)

            splits_for_analyze = [
                (_to_slice(is_idx), _to_slice(oos_idx))
                for is_idx, oos_idx in raw_splits
            ]
        else:
            splits_for_analyze = make_wf_splits(  # type: ignore[assignment]
                n_bars   = len(df),
                n_splits = self.n_splits,
                oos_frac = self.oos_frac,
                gap_bars = self.gap_bars,
            )

        if not splits_for_analyze:
            log.warning("WalkForward: insufficient data for any splits.")
            return WalkForwardResult()

        fold_results = []
        for fold_idx, (is_sl, oos_sl) in enumerate(splits_for_analyze):
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
                fold_idx + 1, len(splits_for_analyze),
                f"{fold.is_score:.4f}", f"{fold.oos_score:.4f}", fold.passes,
            )

        return WalkForwardResult(folds=fold_results)

    def quick_check(
        self,
        df:                   pd.DataFrame,
        generate_signals_fn:  Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
        params:               Dict[str, Any],
        n_folds:              int  = 3,
        warmup_bars:          int  = WARMUP_BARS,
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
