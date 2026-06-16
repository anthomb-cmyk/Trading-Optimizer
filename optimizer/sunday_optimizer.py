"""
APEX Sunday Optimizer — Optuna-powered parameter search.

Runs every Sunday to find the best strategy parameters for the coming week.
Searches up to 500,000 parameter combinations using TPE + Hyperband pruning,
validates with walk-forward analysis, and writes best_params.json.

Usage:
    # From CLI via main.py:
    python main.py optimize --trials 500000

    # Direct:
    python -m optimizer.sunday_optimizer --trials 500000 --symbol MES
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.metrics import compute_score
from backtesting.walk_forward import WalkForwardAnalyzer
from config.logger import get_logger
from config.settings import INSTRUMENTS, OPTIMIZER, OUTPUT_DIR, RISK, WARMUP_BARS
from data.loader import load_bars
from strategies.system import StrategySystem

log = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

BEST_PARAMS_PATH = OUTPUT_DIR / "best_params.json"
STUDY_METRICS_PATH = OUTPUT_DIR / "study_metrics.json"


# ── Objective ────────────────────────────────────────────────────────────────

class _Objective:
    """
    Encapsulates the Optuna objective function.
    Pre-loads all data so it's shared across parallel workers.
    """

    def __init__(
        self,
        symbol:        str,
        timeframe:     str,
        use_wf:        bool = True,
        wf_folds:      int  = 3,     # 3 folds inside objective for speed
        min_trades:    int  = OPTIMIZER["min_trades_per_period"],
    ):
        self.symbol     = symbol
        self.timeframe  = timeframe
        self.use_wf     = use_wf
        self.wf_folds   = wf_folds
        self.min_trades = min_trades
        self.instrument = INSTRUMENTS[symbol]

        # Yahoo V8 API caps intraday history; use the maximum available per TF.
        _TF_MAX_DAYS = {
            "1m": 7, "3m": 60, "5m": 60, "15m": 60, "30m": 60,
            "1h": 730, "4h": 730, "1d": 730, "1w": 730,
        }
        load_days = _TF_MAX_DAYS.get(timeframe, 730)
        log.info("Loading %s %s data for optimizer (%d days)...", symbol, timeframe, load_days)
        self.df = load_bars(symbol, timeframe, days=load_days, use_cache=False)
        log.info("Loaded %d bars (post-warmup: %d)", len(self.df),
                 len(self.df) - WARMUP_BARS)

        self.system  = StrategySystem(symbol)
        self.engine  = BacktestEngine(symbol)
        self.wf      = WalkForwardAnalyzer(self.engine, n_splits=wf_folds)

        # Compute once; used in __call__ and _post_optimize for consistency.
        post_warmup = max(1, len(self.df) - WARMUP_BARS)
        self.min_trades_eff = max(3, min(self.min_trades, post_warmup // 30))
        log.info("min_trades_eff=%d (post-warmup bars: %d)",
                 self.min_trades_eff, post_warmup)

    def __call__(self, trial: optuna.Trial) -> float:
        # ── Sample parameters ─────────────────────────────────────────────────
        params = self.system.get_full_param_space(trial)

        # Persist the FULL param set so _post_optimize can replay exactly.
        # best.params only stores suggest_*-called values; conditional branches
        # that didn't fire are absent, causing the post-opt replay to use
        # strategy defaults instead of the trial's actual values.
        try:
            trial.set_user_attr("full_params", params)
        except Exception:
            pass  # non-critical; _post_optimize falls back to best.params

        # ── Generate signals ──────────────────────────────────────────────────
        try:
            signals = self.system.generate_signals(self.df, params)
        except Exception as exc:
            log.debug("Signal generation failed: %s", exc)
            raise optuna.TrialPruned()

        # ── Quick trade count check before full backtest ───────────────────────
        n_signals = int(signals["long_signal"].sum() + signals["short_signal"].sum())
        if n_signals < self.min_trades_eff:
            raise optuna.TrialPruned()

        # ── Full backtest (IS only for speed during search) ───────────────────
        result = self.engine.run(self.df, signals, params)

        # Use min_trades_eff so Hyperband sees real scores, not always-zero.
        adj_score = compute_score(result.trades, result.equity_curve,
                                  min_trades=self.min_trades_eff)
        trial.report(adj_score, step=max(1, len(result.trades)))

        if trial.should_prune():
            raise optuna.TrialPruned()

        if not result.is_valid(self.min_trades_eff):
            raise optuna.TrialPruned()

        # ── Walk-forward validation on top candidates ─────────────────────────
        if self.use_wf and adj_score > 0.35:
            wf_score = self.wf.quick_check(self.df, signals, params, self.wf_folds)
            final_score = 0.40 * adj_score + 0.60 * wf_score
        else:
            final_score = adj_score

        return float(final_score)


# ── Main optimizer ────────────────────────────────────────────────────────────

class SundayOptimizer:
    """
    Orchestrates the Sunday Optuna optimization run.

    Responsibilities:
      1. Create or resume an Optuna study (SQLite-backed for persistence)
      2. Run trials with TPE sampler + Hyperband pruner
      3. Post-optimization: run full walk-forward on best trial
      4. Save best_params.json for NT8 bridge consumption
      5. Log progress and final summary
    """

    def __init__(
        self,
        symbol:    str = "MES",
        timeframe: str = "15m",
        n_trials:  int = OPTIMIZER["n_trials"],
        n_jobs:    int = OPTIMIZER["n_jobs"],
        timeout_h: float = OPTIMIZER["timeout_hours"],
        use_wf:    bool = True,
    ):
        self.symbol    = symbol
        self.timeframe = timeframe
        self.n_trials  = n_trials
        self.n_jobs    = n_jobs
        self.timeout   = timeout_h * 3600
        self.use_wf    = use_wf

        self.objective = _Objective(symbol, timeframe, use_wf=use_wf)

    def run(self) -> optuna.Study:
        log.info("=" * 60)
        log.info("APEX Sunday Optimizer starting")
        log.info("Symbol: %s | TF: %s | Trials: %s | Jobs: %s | Timeout: %.1fh",
                 self.symbol, self.timeframe, f"{self.n_trials:,}",
                 self.n_jobs, self.timeout / 3600)
        log.info("=" * 60)

        study = self._create_or_load_study()
        start = time.time()

        try:
            study.optimize(
                self.objective,
                n_trials         = self.n_trials,
                timeout          = self.timeout,
                n_jobs           = self.n_jobs,
                gc_after_trial   = True,
                show_progress_bar= sys.stdout.isatty(),
                callbacks        = [self._trial_callback],
            )
        except KeyboardInterrupt:
            log.warning("Optimization interrupted by user.")

        elapsed = time.time() - start
        log.info("Optimization complete in %.1f minutes.", elapsed / 60)
        log.info("Trials: %d completed, %d pruned",
                 len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                 len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))

        self._post_optimize(study)
        return study

    # ── Post-optimization ─────────────────────────────────────────────────────

    def _post_optimize(self, study: optuna.Study):
        """Full walk-forward on best trial + save outputs."""
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            log.error("No successful trials — cannot save best params.")
            return

        best = study.best_trial
        log.info("Best trial #%d | Score=%.6f", best.number, best.value)

        # Use the stored full_params when available; fall back to best.params
        # for studies created before this fix.
        best_params = best.user_attrs.get("full_params") or best.params
        log.info("Replaying with %d params (%s)",
                 len(best_params),
                 "full_params" if "full_params" in best.user_attrs else "best.params fallback")

        # Run full 5-fold WF on the best parameters
        log.info("Running full walk-forward validation on best params...")
        signals = self.objective.system.generate_signals(
            self.objective.df, best_params
        )

        from backtesting.walk_forward import WalkForwardAnalyzer
        wf_full = WalkForwardAnalyzer(
            self.objective.engine,
            n_splits = OPTIMIZER["walk_forward_splits"],
        )
        wf_result = wf_full.analyze(self.objective.df, signals, best_params)
        log.info("Full WF result: %s", wf_result.summary())

        # Final backtest on full data — recompute score with the same
        # min_trades threshold used during optimization for consistency.
        full_result = self.objective.engine.run(
            self.objective.df, signals, best_params
        )
        adj_full_score = compute_score(
            full_result.trades, full_result.equity_curve,
            min_trades=self.objective.min_trades_eff,
        )
        full_result.metrics["score"] = round(adj_full_score, 6)
        full_result.score = adj_full_score
        log.info("Full backtest: %s", full_result.summary())

        # Save outputs
        self._save_best_params(best_params, best.value, full_result.metrics, wf_result)
        self._save_study_metrics(study, full_result, wf_result)

    def _save_best_params(
        self,
        params:     Dict[str, Any],
        score:      float,
        metrics:    Dict[str, float],
        wf_result,
    ):
        output = {
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "symbol":        self.symbol,
            "timeframe":     self.timeframe,
            "score":         round(score, 6),
            "walk_forward":  {
                "is_robust":       wf_result.is_robust,
                "pass_rate":       round(wf_result.pass_rate, 3),
                "avg_oos_score":   round(wf_result.avg_oos_score, 6),
                "avg_degradation": round(wf_result.avg_degradation, 3),
            },
            "performance":   {k: v for k, v in metrics.items()},
            "params":        {
                k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in params.items()
            },
        }

        BEST_PARAMS_PATH.write_text(
            json.dumps(output, indent=2, default=str), encoding="utf-8"
        )
        log.info("Saved best_params.json → %s", BEST_PARAMS_PATH)

    def _save_study_metrics(self, study, full_result, wf_result):
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        scores = [t.value for t in completed if t.value is not None]

        summary = {
            "generated_at":       datetime.now(timezone.utc).isoformat(),
            "total_trials":       len(study.trials),
            "completed_trials":   len(completed),
            "pruned_trials":      len(study.trials) - len(completed),
            "best_score":         round(study.best_value, 6) if study.best_value else 0,
            "score_p25":          round(float(np.percentile(scores, 25)), 6) if scores else 0,
            "score_p50":          round(float(np.percentile(scores, 50)), 6) if scores else 0,
            "score_p75":          round(float(np.percentile(scores, 75)), 6) if scores else 0,
            "walk_forward_robust": wf_result.is_robust,
            "full_backtest":      full_result.metrics,
        }
        STUDY_METRICS_PATH.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        log.info("Saved study_metrics.json → %s", STUDY_METRICS_PATH)

    # ── Study factory ─────────────────────────────────────────────────────────

    def _create_or_load_study(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials            = 200,   # random exploration before TPE
            multivariate                = True,  # model parameter correlations
            warn_independent_sampling   = False, # suppress dynamic-space warnings
            seed                        = 42,
        )
        pruner = optuna.pruners.HyperbandPruner(
            min_resource    = 3,        # min trades before pruning eligible
            max_resource    = 200,
            reduction_factor= 3,
        )

        study = optuna.create_study(
            study_name   = f"{OPTIMIZER['study_name']}_{self.symbol}_{self.timeframe}",
            storage      = OPTIMIZER["storage"],
            direction    = "maximize",
            sampler      = sampler,
            pruner       = pruner,
            load_if_exists = True,
        )

        n_existing = len(study.trials)
        if n_existing > 0:
            log.info("Resuming study with %d existing trials.", n_existing)
        else:
            log.info("Starting fresh study.")

        return study

    # ── Progress callback ─────────────────────────────────────────────────────

    @staticmethod
    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        n = len([t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE])
        if n > 0 and n % 1000 == 0:
            log.info(
                "Progress: %6d trials | Best=%.6f (trial #%d)",
                n, study.best_value, study.best_trial.number,
            )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="APEX Sunday Optimizer")
    parser.add_argument("--symbol",    default="MES",    choices=["MES", "MNQ"])
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--trials",    type=int, default=OPTIMIZER["n_trials"])
    parser.add_argument("--jobs",      type=int, default=OPTIMIZER["n_jobs"])
    parser.add_argument("--timeout",   type=float, default=OPTIMIZER["timeout_hours"],
                        help="Timeout in hours")
    parser.add_argument("--no-wf",     action="store_true",
                        help="Skip walk-forward validation (faster, less robust)")
    args = parser.parse_args()

    optimizer = SundayOptimizer(
        symbol    = args.symbol,
        timeframe = args.timeframe,
        n_trials  = args.trials,
        n_jobs    = args.jobs,
        timeout_h = args.timeout,
        use_wf    = not args.no_wf,
    )
    optimizer.run()


if __name__ == "__main__":
    main()
