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

Overfitting-aware validation (T4.x):
  T4.3 - Trial budget: recommended_max_trials() caps n_trials to a data-derived
         bound. If configured trials exceed the recommendation, a WARN is emitted.
  T4.4 - Locked holdout: the most recent holdout_frac of bars are split off
         before optimization. The objective never sees holdout rows. After the
         study the single best config is evaluated once on the holdout.
  Scoring: objective uses compute_score_v2 (risk-adjusted, DSR-aware).
  Logging: start_run / finish_run / log_optimization wired to Supabase.
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
from backtesting.metrics import (
    compute_score_v2,
    deflated_sharpe_ratio,
    pbo_from_oos_matrix,
    per_observation_sharpe_ratio,
    sharpe_ratio,
)
from backtesting.walk_forward import WalkForwardAnalyzer, recommended_max_trials
from config import run_logger as rl
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

    T4.4 - Holdout isolation: ``df_train`` is the slice the objective can see.
    The ``df_holdout`` portion is never passed to __call__ -- it is evaluated
    once after the study by ``SundayOptimizer._post_optimize``.
    """

    def __init__(
        self,
        symbol:         str,
        timeframe:      str,
        use_wf:         bool  = True,
        wf_folds:       int   = 3,       # 3 folds inside objective for speed
        min_trades:     int   = OPTIMIZER["min_trades_per_period"],
        holdout_frac:   float = 0.20,    # T4.4: fraction held out
    ):
        self.symbol       = symbol
        self.timeframe    = timeframe
        self.use_wf       = use_wf
        self.wf_folds     = wf_folds
        self.min_trades   = min_trades
        self.holdout_frac = holdout_frac
        self.instrument   = INSTRUMENTS[symbol]

        # Yahoo V8 API caps intraday history; use the maximum available per TF.
        _TF_MAX_DAYS = {
            "1m": 7, "3m": 60, "5m": 60, "15m": 60, "30m": 60,
            "1h": 730, "4h": 730, "1d": 730, "1w": 730,
        }
        load_days = _TF_MAX_DAYS.get(timeframe, 730)
        log.info("Loading %s %s data for optimizer (%d days)...", symbol, timeframe, load_days)
        self._df_full = load_bars(symbol, timeframe, days=load_days, use_cache=False)
        log.info("Loaded %d bars total", len(self._df_full))

        # ── T4.4: Holdout split ───────────────────────────────────────────────
        n_total        = len(self._df_full)
        holdout_n      = max(1, int(n_total * holdout_frac))
        self.holdout_start_idx = n_total - holdout_n   # first holdout bar

        # Training slice: first (1 - holdout_frac) of bars
        self.df = self._df_full.iloc[: self.holdout_start_idx].copy()
        # Holdout slice: last holdout_frac of bars (NEVER seen by objective)
        self.df_holdout = self._df_full.iloc[self.holdout_start_idx :].copy()

        log.info(
            "Holdout split: train=%d bars, holdout=%d bars (last %.0f%%)",
            len(self.df), len(self.df_holdout), holdout_frac * 100,
        )

        self.system  = StrategySystem(symbol)
        self.engine  = BacktestEngine(symbol)
        self.wf      = WalkForwardAnalyzer(self.engine, n_splits=wf_folds)

        # Compute once; used in __call__ and _post_optimize for consistency.
        post_warmup = max(1, len(self.df) - WARMUP_BARS)
        self.min_trades_eff = max(3, min(self.min_trades, post_warmup // 30))
        log.info("min_trades_eff=%d (post-warmup bars: %d)",
                 self.min_trades_eff, post_warmup)

        # Track effective trial count for DSR computation (T4.3)
        self._effective_trials: int = 0

    def __call__(self, trial: optuna.Trial) -> float:
        self._effective_trials += 1

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

        # ── Generate signals on TRAINING data only (T4.4) ─────────────────────
        try:
            signals = self.system.generate_signals(self.df, params)
        except Exception as exc:
            log.debug("Signal generation failed: %s", exc)
            raise optuna.TrialPruned()

        # ── Quick trade count check before full backtest ───────────────────────
        n_signals = int(signals["long_signal"].sum() + signals["short_signal"].sum())
        if n_signals < self.min_trades_eff:
            raise optuna.TrialPruned()

        # ── Full backtest on TRAINING data (T4.4) ─────────────────────────────
        result = self.engine.run(self.df, signals, params)

        # T4.3: use compute_score_v2 with effective trial count so DSR penalty
        # is included in the objective signal seen by TPE.
        adj_score = compute_score_v2(
            result.trades, result.equity_curve,
            n_trials  = max(1, self._effective_trials),
            min_trades= self.min_trades_eff,
        )
        trial.report(adj_score, step=max(1, len(result.trades)))

        if trial.should_prune():
            raise optuna.TrialPruned()

        if not result.is_valid(self.min_trades_eff):
            raise optuna.TrialPruned()

        # ── Walk-forward validation on top candidates ─────────────────────────
        # Pass the generate_signals callable so walk-forward generates signals
        # per fold (causal), avoiding look-ahead from full-dataset pre-computation.
        # WF is run on self.df (training only) — never on holdout rows.
        if self.use_wf and adj_score > 0.35:
            wf_score = self.wf.quick_check(
                self.df, self.system.generate_signals, params, self.wf_folds,
            )
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
      4. Evaluate best config ONCE on locked holdout (T4.4)
      5. Compute Deflated Sharpe and PBO (T4.3)
      6. Save best_params.json for NT8 bridge consumption
      7. Log progress and final summary to Supabase (T4.3 / T4.4)
    """

    def __init__(
        self,
        symbol:       str   = "MES",
        timeframe:    str   = "15m",
        n_trials:     int   = OPTIMIZER["n_trials"],
        n_jobs:       int   = OPTIMIZER["n_jobs"],
        timeout_h:    float = OPTIMIZER["timeout_hours"],
        use_wf:       bool  = True,
        holdout_frac: float = 0.20,
    ):
        self.symbol       = symbol
        self.timeframe    = timeframe
        self.n_trials     = n_trials
        self.n_jobs       = n_jobs
        self.timeout      = timeout_h * 3600
        self.use_wf       = use_wf
        self.holdout_frac = holdout_frac

        self.objective = _Objective(
            symbol, timeframe, use_wf=use_wf, holdout_frac=holdout_frac,
        )

    def run(self) -> optuna.Study:
        log.info("=" * 60)
        log.info("APEX Sunday Optimizer starting")
        log.info("Symbol: %s | TF: %s | Trials: %s | Jobs: %s | Timeout: %.1fh",
                 self.symbol, self.timeframe, f"{self.n_trials:,}",
                 self.n_jobs, self.timeout / 3600)

        # ── T4.3: Trial budget governance ─────────────────────────────────────
        n_oos_obs = len(self.objective.df_holdout)
        rec_max   = recommended_max_trials(n_oos_obs)
        log.info(
            "Trial budget: configured=%d, recommended_max=%d (n_oos_obs=%d)",
            self.n_trials, rec_max, n_oos_obs,
        )
        if self.n_trials > rec_max:
            rl.log_event(
                f"Configured n_trials={self.n_trials:,} exceeds data-derived "
                f"recommendation of {rec_max:,} (n_oos_obs={n_oos_obs}). "
                "Deflated Sharpe will penalise excess trials automatically.",
                level="WARN",
                kind="validation",
                payload={
                    "n_trials_configured": self.n_trials,
                    "n_trials_recommended": rec_max,
                    "n_oos_obs": n_oos_obs,
                },
            )
        log.info("=" * 60)

        # ── Supabase run envelope (T4.3) ──────────────────────────────────────
        run_uuid = rl.start_run(
            "optimize",
            symbol    = self.symbol,
            timeframe = self.timeframe,
            config    = {
                "n_trials":       self.n_trials,
                "n_jobs":         self.n_jobs,
                "holdout_frac":   self.holdout_frac,
                "rec_max_trials": rec_max,
            },
        )

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

        self._post_optimize(study, run_uuid=run_uuid)
        rl.finish_run(run_uuid, status="ok")
        return study

    # ── Post-optimization ─────────────────────────────────────────────────────

    def _post_optimize(self, study: optuna.Study, *, run_uuid: Optional[str] = None):
        """
        Full walk-forward on best trial + locked holdout evaluation + Supabase
        logging of honesty metrics.

        T4.3: Effective trial count drives Deflated Sharpe (DSR) and PBO.
        T4.4: Best config is evaluated ONCE on df_holdout.
        """
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            log.error("No successful trials — cannot save best params.")
            rl.finish_run(run_uuid or "", status="no_trials") if run_uuid else None
            return

        best = study.best_trial
        log.info("Best trial #%d | Score=%.6f", best.number, best.value)

        # Use the stored full_params when available; fall back to best.params
        # for studies created before this fix.
        best_params = best.user_attrs.get("full_params") or best.params
        log.info("Replaying with %d params (%s)",
                 len(best_params),
                 "full_params" if "full_params" in best.user_attrs else "best.params fallback")

        effective_trials = max(1, self.objective._effective_trials)
        log.info("Effective trials run: %d", effective_trials)

        # ── Walk-forward on TRAINING data ─────────────────────────────────────
        log.info("Running full walk-forward validation on best params (train only)...")
        wf_full = WalkForwardAnalyzer(
            self.objective.engine,
            n_splits = OPTIMIZER["walk_forward_splits"],
        )
        wf_result = wf_full.analyze(
            self.objective.df,
            self.objective.system.generate_signals,
            best_params,
        )
        log.info("Full WF result: %s", wf_result.summary())

        # OOS score = avg across WF folds (the training-side OOS windows)
        oos_score = wf_result.avg_oos_score

        # ── Full backtest on TRAINING data (summary diagnostic) ───────────────
        signals = self.objective.system.generate_signals(
            self.objective.df, best_params
        )
        full_result = self.objective.engine.run(
            self.objective.df, signals, best_params
        )
        adj_full_score = compute_score_v2(
            full_result.trades, full_result.equity_curve,
            n_trials  = effective_trials,
            min_trades= self.objective.min_trades_eff,
        )
        full_result.metrics["score"] = round(adj_full_score, 6)
        full_result.score = adj_full_score
        log.info("Train backtest: %s", full_result.summary())

        # ── T4.3: Deflated Sharpe (DSR) using effective trials ────────────────
        # DSR units contract: deflated_sharpe_ratio multiplies observed_sr by
        # sqrt(n_obs - 1) internally, so it must be fed a PER-OBSERVATION (non-
        # annualized) Sharpe together with the number of those return
        # observations.  Feeding the ANNUALIZED Sharpe (which already embeds a
        # sqrt(periods_per_year) factor) with a large n_obs double-counts the
        # time scaling and saturates Phi(...) to ~1.0 even on a zero-edge run --
        # the false-confidence guardrail bug.  The annualized Sharpe is still
        # logged for human reading.
        sr_per_obs, n_obs = per_observation_sharpe_ratio(full_result.equity_curve)
        sr_annualized     = sharpe_ratio(full_result.equity_curve)
        dsr               = deflated_sharpe_ratio(
            observed_sr = sr_per_obs,
            n_trials    = effective_trials,
            n_obs       = n_obs,
        )
        log.info(
            "Deflated Sharpe: %.4f (SR_per_obs=%.4f, SR_annualized=%.4f, "
            "n_trials=%d, n_obs=%d)",
            dsr, sr_per_obs, sr_annualized, effective_trials, n_obs,
        )

        # ── T4.3: PBO via a GENUINE per-period OOS performance matrix ──────────
        # A real CSCV/PBO needs a (T_observations x N_configs) matrix of
        # per-period OUT-OF-SAMPLE scores, NOT a single summary score per trial
        # tiled across fake time blocks.  Tiling makes every row identical, so
        # the IS-best is mechanically the OOS-best and PBO collapses to 0.0 by
        # construction -- a false-confidence guardrail bug.
        #
        # Here we replay walk-forward on the top completed trials and stack each
        # trial's PER-FOLD OOS scores as one column.  Rows = WF folds (genuine
        # OOS observations), columns = configs.  If too few trials produce
        # enough comparable OOS folds, PBO is reported as None (undefined)
        # rather than a misleading 0.0.
        pbo = self._compute_oos_pbo(completed, best_params)
        if pbo is None:
            log.info(
                "Probability of Backtest Overfitting (PBO): UNDEFINED "
                "(too few comparable OOS trials to build a genuine matrix)"
            )
        else:
            log.info("Probability of Backtest Overfitting (PBO): %.4f", pbo)

        # ── T4.4: Locked holdout evaluation ───────────────────────────────────
        holdout_score = 0.0
        if len(self.objective.df_holdout) > 0:
            log.info(
                "Evaluating best config on HOLDOUT (%d bars, never seen by optimizer)...",
                len(self.objective.df_holdout),
            )
            try:
                ho_signals = self.objective.system.generate_signals(
                    self.objective.df_holdout, best_params
                )
                ho_result = self.objective.engine.run(
                    self.objective.df_holdout, ho_signals, best_params
                )
                holdout_score = compute_score_v2(
                    ho_result.trades, ho_result.equity_curve,
                    n_trials  = effective_trials,
                    min_trades= self.objective.min_trades_eff,
                )
                log.info(
                    "Holdout result: trades=%d score=%.4f",
                    len(ho_result.trades), holdout_score,
                )
                # Log holdout backtest to Supabase
                rl.log_backtest(
                    symbol        = self.symbol,
                    timeframe     = self.timeframe,
                    run_uuid      = run_uuid,
                    strategy_name = "order_block",
                    is_holdout    = True,
                    params        = best_params,
                    metrics       = {
                        **ho_result.metrics,
                        "deflated_sharpe": dsr,
                        "pbo":             pbo,
                        "effective_trials": effective_trials,
                        "holdout_score":    holdout_score,
                    },
                )
            except Exception as exc:
                log.error("Holdout evaluation failed: %s", exc)
        else:
            log.warning("Holdout set is empty; skipping holdout evaluation.")

        # ── Supabase optimization log (T4.3) ──────────────────────────────────
        rl.log_optimization(
            symbol           = self.symbol,
            timeframe        = self.timeframe,
            strategy_name    = "order_block",
            run_uuid         = run_uuid,
            n_trials         = len(study.trials),
            best_score       = float(best.value) if best.value is not None else 0.0,
            best_params      = best_params,
            deflated_sharpe  = float(dsr),
            pbo              = (float(pbo) if pbo is not None else None),
            effective_trials = int(effective_trials),
            oos_score        = float(oos_score),
            holdout_score    = float(holdout_score),
            status           = "ok",
        )

        # ── Save outputs ──────────────────────────────────────────────────────
        self._save_best_params(
            best_params, best.value, full_result.metrics, wf_result,
            dsr=dsr, pbo=pbo, holdout_score=holdout_score,
            effective_trials=effective_trials,
        )
        self._save_study_metrics(study, full_result, wf_result)

    # ── PBO from genuine per-fold OOS matrix ──────────────────────────────────

    def _compute_oos_pbo(
        self,
        completed,
        best_params: Dict[str, Any],
        *,
        max_configs: int = 24,
        min_configs: int = 4,
    ) -> Optional[float]:
        """
        Build a genuine (T_folds x N_configs) per-period OOS performance matrix
        and compute PBO from it.

        For each of up to ``max_configs`` top completed trials we replay full
        walk-forward (train data only) and collect the per-fold OOS scores.
        Trials are kept only if they yield the SAME number of OOS folds as the
        majority, so every column is aligned to the same time blocks.  Each
        column is one config; each row is one OOS fold (a real out-of-sample
        observation).

        Returns a float in [0, 1], or ``None`` when too few comparable trials
        exist to form a genuine matrix (reported as undefined, never 0.0).
        """
        # Rank completed trials by objective value; take the strongest configs.
        scored = sorted(
            (t for t in completed if t.value is not None),
            key=lambda t: t.value,
            reverse=True,
        )
        if len(scored) < min_configs:
            return None
        scored = scored[:max_configs]

        wf = WalkForwardAnalyzer(
            self.objective.engine,
            n_splits=OPTIMIZER["walk_forward_splits"],
        )

        columns: list = []   # each entry: list of per-fold OOS scores for a config
        for t in scored:
            params = t.user_attrs.get("full_params") or t.params
            try:
                res = wf.analyze(
                    self.objective.df,
                    self.objective.system.generate_signals,
                    params,
                )
            except Exception as exc:   # noqa: BLE001 - one bad config must not abort PBO
                log.debug("PBO: WF replay failed for trial #%d: %s", t.number, exc)
                continue
            fold_scores = [f.oos_score for f in res.folds]
            if fold_scores:
                columns.append(fold_scores)

        if len(columns) < min_configs:
            return None

        # Align all configs to the most common fold count so the matrix is
        # rectangular and every row is the same OOS time block across configs.
        from collections import Counter
        n_folds = Counter(len(c) for c in columns).most_common(1)[0][0]
        aligned = [c for c in columns if len(c) == n_folds]
        if len(aligned) < min_configs:
            return None

        # perf_matrix: rows = OOS folds (observations), cols = configs.
        perf_matrix = np.array(aligned, dtype=float).T   # (n_folds, n_configs)
        return pbo_from_oos_matrix(perf_matrix)

    def _save_best_params(
        self,
        params:           Dict[str, Any],
        score:            float,
        metrics:          Dict[str, float],
        wf_result,
        *,
        dsr:              float          = 0.0,
        pbo:              Optional[float] = 0.0,
        holdout_score:    float          = 0.0,
        effective_trials: int            = 0,
    ):
        output = {
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "symbol":           self.symbol,
            "timeframe":        self.timeframe,
            "score":            round(score, 6),
            "holdout_score":    round(holdout_score, 6),
            "deflated_sharpe":  round(dsr, 6),
            "pbo":              (round(pbo, 6) if pbo is not None else None),
            "effective_trials": int(effective_trials),
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
    parser.add_argument("--symbol",       default="MES",    choices=["MES", "MNQ"])
    parser.add_argument("--timeframe",    default="15m")
    parser.add_argument("--trials",       type=int,   default=OPTIMIZER["n_trials"])
    parser.add_argument("--jobs",         type=int,   default=OPTIMIZER["n_jobs"])
    parser.add_argument("--timeout",      type=float, default=OPTIMIZER["timeout_hours"],
                        help="Timeout in hours")
    parser.add_argument("--no-wf",        action="store_true",
                        help="Skip walk-forward validation (faster, less robust)")
    parser.add_argument("--holdout-frac", type=float, default=0.20,
                        help="Fraction of bars reserved as a locked holdout (default 0.20)")
    args = parser.parse_args()

    optimizer = SundayOptimizer(
        symbol       = args.symbol,
        timeframe    = args.timeframe,
        n_trials     = args.trials,
        n_jobs       = args.jobs,
        timeout_h    = args.timeout,
        use_wf       = not args.no_wf,
        holdout_frac = args.holdout_frac,
    )
    optimizer.run()


if __name__ == "__main__":
    main()
