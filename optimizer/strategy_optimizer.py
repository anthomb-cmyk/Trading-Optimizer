"""
APEX Single-Strategy Honest Optimizer.

Mirrors the SundayOptimizer's proven structure (locked holdout, per-fold
causal walk-forward, compute_score_v2 objective, FIXED DSR/PBO wiring,
run_logger calls) but targets ONE strategy at a time rather than the full
StrategySystem.

Usage:
    # Via main.py CLI:
    python main.py optimize-strategy --strategy short_term_reversal \\
        --symbol MES --timeframe 15m --trials 20 --jobs 1 --timeout 0.05

    # Direct:
    from optimizer.strategy_optimizer import StrategyOptimizer
    result = StrategyOptimizer("opening_range_breakout").run()

Overfitting guardrails (identical to SundayOptimizer T4.x):
  T4.3 - n_trials defaults to recommended_max_trials(n_oos_obs); budget
          warning emitted when configured trials exceed recommendation.
  T4.4 - Last holdout_frac of bars locked off before optimization; the
          objective NEVER sees holdout rows.  Best config evaluated ONCE
          on holdout after the study.
  DSR  - Fed the PER-OBSERVATION (non-annualized) Sharpe + matching n_obs
          via per_observation_sharpe_ratio, preventing false-confidence
          saturation on zero-edge strategies.
  PBO  - Built from a genuine (n_folds x n_configs) per-fold OOS matrix;
          never from tiled single-scalar columns.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
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
from backtesting.walk_forward import WalkForwardAnalyzer, generate_fold_signals, recommended_max_trials
from config import run_logger as rl
from config.logger import get_logger
from config.settings import INSTRUMENTS, OPTIMIZER, OUTPUT_DIR, WARMUP_BARS
from data.loader import load_bars
from strategies import get_strategy

log = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Yahoo timeframe caps (mirrors sunday_optimizer) ──────────────────────────
_TF_MAX_DAYS: Dict[str, int] = {
    "1m": 7, "3m": 60, "5m": 60, "15m": 60, "30m": 60,
    "1h": 730, "4h": 730, "1d": 730, "1w": 730,
}

# ── Trade-frequency annualization basis ──────────────────────────────────────
# Used to convert a raw trade count over a scored window into trades-per-year so
# the optimizer cannot hide in degenerate low-trade configs (the prior holdout
# was under-powered at 3-5 trades). Basis: equity-index futures (MES/MNQ/...)
# trade ~64 RTH 15m bars per session (09:30-16:00 = 6.5h = 26 15m bars; the
# extended-hours/globex envelope this loader fetches widens that to ~64 bars/day)
# across ~252 trading days per year -> 64 * 252 = 16128 bars/year. This is the
# per-bar denominator; on coarser timeframes the strategy simply produces fewer
# bars per window, so trades_per_year scales down naturally and the same
# min_trades_per_year floor still applies.
BARS_PER_YEAR: int = 64 * 252  # ≈ 16128


# ── Objective ─────────────────────────────────────────────────────────────────

class _StrategyObjective:
    """
    Optuna objective for a single strategy.

    T4.4 - Holdout isolation: ``df_train`` is the only slice the objective
    can see.  ``df_holdout`` is passed through but never touched during
    optimization; it is evaluated ONCE by StrategyOptimizer._post_optimize.
    """

    def __init__(
        self,
        strategy_name:       str,
        symbol:              str,
        timeframe:           str,
        use_wf:              bool  = True,
        wf_folds:            int   = 3,
        min_trades:          int   = OPTIMIZER["min_trades_per_period"],
        holdout_frac:        float = 0.20,
        min_trades_per_year: float = 50.0,
        days:                Optional[int] = None,
        use_cache:           bool  = True,
    ):
        self.strategy_name       = strategy_name
        self.symbol              = symbol
        self.timeframe           = timeframe
        self.use_wf              = use_wf
        self.wf_folds            = wf_folds
        self.min_trades          = min_trades
        self.holdout_frac        = holdout_frac
        self.min_trades_per_year = float(min_trades_per_year)

        # Instantiate the strategy (reads symbol for instrument config)
        self.strategy = get_strategy(strategy_name, symbol)

        # Load bars
        load_days = days if days is not None else _TF_MAX_DAYS.get(timeframe, 730)
        log.info(
            "Loading %s %s data for strategy optimizer (%d days, cache=%s)...",
            symbol, timeframe, load_days, use_cache,
        )
        self._df_full = load_bars(symbol, timeframe, days=load_days, use_cache=use_cache)
        log.info("Loaded %d bars total", len(self._df_full))

        # ── T4.4: Holdout split ───────────────────────────────────────────────
        n_total               = len(self._df_full)
        holdout_n             = max(1, int(n_total * holdout_frac))
        self.holdout_start_idx = n_total - holdout_n

        self.df         = self._df_full.iloc[: self.holdout_start_idx].copy()  # train
        self.df_holdout = self._df_full.iloc[self.holdout_start_idx :].copy()  # locked

        log.info(
            "Holdout split: train=%d bars, holdout=%d bars (last %.0f%%)",
            len(self.df), len(self.df_holdout), holdout_frac * 100,
        )

        self.engine = BacktestEngine(symbol)
        self.wf     = WalkForwardAnalyzer(self.engine, n_splits=wf_folds)

        # Effective min_trades accounting for warmup
        post_warmup          = max(1, len(self.df) - WARMUP_BARS)
        self.min_trades_eff  = max(3, min(self.min_trades, post_warmup // 30))
        log.info(
            "min_trades_eff=%d (post-warmup bars: %d)",
            self.min_trades_eff, post_warmup,
        )

        # Trial counter for DSR penalty inside the objective (T4.3)
        self._effective_trials: int = 0

    def __call__(self, trial: optuna.Trial) -> float:
        self._effective_trials += 1

        # ── Sample from the strategy's own param space ────────────────────────
        params = self.strategy.get_param_space(trial)

        # Persist full param set for replay in _post_optimize
        try:
            trial.set_user_attr("full_params", params)
        except Exception:
            pass

        # ── Generate signals on TRAINING data only (T4.4) ─────────────────────
        try:
            signals = self.strategy.generate_signals(self.df, params)
        except Exception as exc:
            log.debug("Signal generation failed: %s", exc)
            raise optuna.TrialPruned()

        n_signals = int(signals["long_signal"].sum() + signals["short_signal"].sum())
        if n_signals < self.min_trades_eff:
            raise optuna.TrialPruned()

        # ── Full backtest on TRAINING data ────────────────────────────────────
        result = self.engine.run(self.df, signals, params)

        adj_score = compute_score_v2(
            result.trades, result.equity_curve,
            n_trials   = max(1, self._effective_trials),
            min_trades = self.min_trades_eff,
        )
        trial.report(adj_score, step=max(1, len(result.trades)))

        if trial.should_prune():
            raise optuna.TrialPruned()

        if not result.is_valid(self.min_trades_eff):
            raise optuna.TrialPruned()

        # ── Walk-forward validation (causal, per-fold signals) ────────────────
        # Runs on self.df (training only) — holdout rows never touched.
        if self.use_wf and adj_score > 0.35:
            wf_score = self.wf.quick_check(
                self.df,
                self.strategy.generate_signals,
                params,
                self.wf_folds,
            )
            final_score = 0.40 * adj_score + 0.60 * wf_score
        else:
            final_score = adj_score

        # ── Minimum-trade-frequency penalty ───────────────────────────────────
        # Annualize the trade count over the scored window so the search cannot
        # hide in degenerate low-trade configs (the prior holdout was under-
        # powered at 3-5 trades). n_scored_bars is the training window the score
        # was computed over; BARS_PER_YEAR (=64*252≈16128) is the documented
        # per-bar annualization basis (see module top).
        n_scored_bars  = max(1, len(self.df))
        total_trades   = len(result.trades)
        trades_per_year = total_trades / (n_scored_bars / BARS_PER_YEAR)
        trial.set_user_attr("trades_per_year", float(trades_per_year))

        # Smooth linear penalty toward 0 for configs that trade too rarely:
        # below the floor the score is scaled by (tpy / min_tpy); at/above the
        # floor it is untouched. Pushes the optimizer to configs that trade
        # often enough to validate.
        if self.min_trades_per_year > 0 and trades_per_year < self.min_trades_per_year:
            final_score *= trades_per_year / self.min_trades_per_year

        return float(final_score)


# ── Main optimizer ─────────────────────────────────────────────────────────────

class StrategyOptimizer:
    """
    Single-strategy Optuna optimizer.

    Mirrors SundayOptimizer's proven structure:
      1. TPE study (SQLite-backed)
      2. Per-fold causal walk-forward objective
      3. Post-study: full WF on best trial (train only)
      4. T4.4 locked holdout: best config evaluated ONCE on holdout
      5. T4.3 FIXED DSR (per-observation SR) + genuine per-fold PBO matrix
      6. run_logger calls for Supabase audit trail
      7. output/strategy_best_<name>.json with headline results
    """

    def __init__(
        self,
        strategy_name:       str,
        symbol:              str   = "MES",
        timeframe:           str   = "15m",
        n_trials:            Optional[int]   = None,   # None → recommended_max_trials
        n_jobs:              int   = 1,
        timeout_h:           Optional[float] = None,
        holdout_frac:        float = 0.20,
        use_wf:              bool  = True,
        min_trades_per_year: float = 50.0,
        days:                Optional[int] = None,
        use_cache:           bool  = True,
    ):
        self.strategy_name       = strategy_name
        self.symbol              = symbol
        self.timeframe           = timeframe
        self.n_jobs              = n_jobs
        self.timeout             = (timeout_h or 0.0) * 3600  # 0 → no timeout
        self.holdout_frac        = holdout_frac
        self.use_wf              = use_wf
        self.min_trades_per_year = float(min_trades_per_year)

        self.objective = _StrategyObjective(
            strategy_name       = strategy_name,
            symbol              = symbol,
            timeframe           = timeframe,
            use_wf              = use_wf,
            holdout_frac        = holdout_frac,
            min_trades_per_year = min_trades_per_year,
            days                = days,
            use_cache           = use_cache,
        )

        # Default n_trials: data-derived recommendation (T4.3), NOT 500k
        n_oos_obs    = len(self.objective.df_holdout)
        rec_max      = recommended_max_trials(n_oos_obs)
        self.n_trials = n_trials if n_trials is not None else rec_max
        self._rec_max = rec_max

    def run(self) -> Dict[str, Any]:
        log.info("=" * 60)
        log.info(
            "StrategyOptimizer: %s | %s %s | Trials=%d | Jobs=%d | Timeout=%.2fh",
            self.strategy_name, self.symbol, self.timeframe,
            self.n_trials, self.n_jobs, self.timeout / 3600,
        )

        # ── T4.3: Trial budget governance ─────────────────────────────────────
        n_oos_obs = len(self.objective.df_holdout)
        log.info(
            "Trial budget: configured=%d, recommended_max=%d (n_oos_obs=%d)",
            self.n_trials, self._rec_max, n_oos_obs,
        )
        if self.n_trials > self._rec_max:
            rl.log_event(
                f"[{self.strategy_name}] n_trials={self.n_trials:,} exceeds "
                f"data-derived recommendation of {self._rec_max:,} "
                f"(n_oos_obs={n_oos_obs}). DSR will penalise excess trials.",
                level="WARN",
                kind="validation",
                payload={
                    "strategy":             self.strategy_name,
                    "n_trials_configured":  self.n_trials,
                    "n_trials_recommended": self._rec_max,
                    "n_oos_obs":            n_oos_obs,
                },
            )
        log.info("=" * 60)

        # ── Supabase run envelope ─────────────────────────────────────────────
        run_uuid = rl.start_run(
            "optimize",
            symbol    = self.symbol,
            timeframe = self.timeframe,
            config    = {
                "strategy":       self.strategy_name,
                "n_trials":       self.n_trials,
                "n_jobs":         self.n_jobs,
                "holdout_frac":   self.holdout_frac,
                "rec_max_trials": self._rec_max,
            },
        )

        study = self._create_study()
        start = time.time()

        try:
            study.optimize(
                self.objective,
                n_trials          = self.n_trials,
                timeout           = self.timeout if self.timeout > 0 else None,
                n_jobs            = self.n_jobs,
                gc_after_trial    = True,
                show_progress_bar = sys.stdout.isatty(),
                callbacks         = [self._trial_callback],
            )
        except KeyboardInterrupt:
            log.warning("Optimization interrupted by user.")

        elapsed = time.time() - start
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        log.info(
            "Optimization complete in %.1f min. Trials: %d completed, %d pruned",
            elapsed / 60, len(completed), len(pruned),
        )

        result = self._post_optimize(study, run_uuid=run_uuid)
        rl.finish_run(run_uuid, status="ok")
        return result

    # ── Post-optimization ──────────────────────────────────────────────────────

    def _post_optimize(
        self,
        study: optuna.Study,
        *,
        run_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full WF on best trial + locked holdout evaluation + Supabase logging.

        DSR wiring (guardrail fix):
          per_observation_sharpe_ratio returns (sr_per_obs, n_obs) on the same
          per-period basis.  deflated_sharpe_ratio is fed that pair, avoiding
          the double-scaling that makes Phi(...) saturate to 1.0 on zero-edge
          strategies when an annualized SR is fed with a large n_obs.

        PBO wiring (guardrail fix):
          Genuine (n_folds x n_configs) matrix built by replaying walk-forward
          on the top completed trials.  Never a tiled single-scalar per trial.
        """
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            log.error("No successful trials — cannot compute results.")
            if run_uuid:
                rl.finish_run(run_uuid, status="no_trials")
            return {"status": "no_trials"}

        best        = study.best_trial
        best_params = best.user_attrs.get("full_params") or best.params
        log.info(
            "Best trial #%d | Score=%.6f | Params from %s",
            best.number, best.value,
            "full_params" if "full_params" in best.user_attrs else "best.params fallback",
        )

        effective_trials = max(1, self.objective._effective_trials)
        log.info("Effective trials: %d", effective_trials)

        # ── Full walk-forward on TRAINING data ─────────────────────────────────
        log.info("Running full walk-forward on best params (train only)...")
        wf_full   = WalkForwardAnalyzer(
            self.objective.engine,
            n_splits = OPTIMIZER["walk_forward_splits"],
        )
        wf_result = wf_full.analyze(
            self.objective.df,
            self.objective.strategy.generate_signals,
            best_params,
        )
        log.info("Full WF: %s", wf_result.summary())
        oos_score = wf_result.avg_oos_score

        # ── Full backtest on TRAINING data (diagnostic) ────────────────────────
        signals     = self.objective.strategy.generate_signals(self.objective.df, best_params)
        train_result = self.objective.engine.run(self.objective.df, signals, best_params)
        adj_train   = compute_score_v2(
            train_result.trades, train_result.equity_curve,
            n_trials   = effective_trials,
            min_trades = self.objective.min_trades_eff,
        )
        train_result.metrics["score"] = round(adj_train, 6)

        # ── Trade frequency (annualized) on the training window ────────────────
        # Same basis as the in-objective penalty: total_trades over the scored
        # bars, annualized by BARS_PER_YEAR (=64*252≈16128). Surfaced so a
        # config that passed only by trading rarely is visible in the headline.
        n_train_bars    = max(1, len(self.objective.df))
        trades_per_year = len(train_result.trades) / (n_train_bars / BARS_PER_YEAR)
        train_result.metrics["trades_per_year"] = round(trades_per_year, 2)
        log.info(
            "Trade frequency: %.1f trades/year (%d trades over %d train bars; "
            "min_trades_per_year=%.0f)",
            trades_per_year, len(train_result.trades), n_train_bars,
            self.min_trades_per_year,
        )
        log.info("Train backtest: %s", train_result.summary())

        # ── T4.3: FIXED Deflated Sharpe (per-observation basis) ───────────────
        # DSR must be fed per_observation_sharpe_ratio, not the annualized SR.
        # Feeding the annualized SR to deflated_sharpe_ratio with a large n_obs
        # double-counts the sqrt(periods_per_year) scaling, saturating Phi to
        # ~1.0 even on a zero-edge strategy (the false-confidence guardrail bug).
        sr_per_obs, n_obs = per_observation_sharpe_ratio(train_result.equity_curve)
        sr_annualized     = sharpe_ratio(train_result.equity_curve)
        dsr               = deflated_sharpe_ratio(
            observed_sr = sr_per_obs,
            n_trials    = effective_trials,
            n_obs       = n_obs,
        )
        log.info(
            "Deflated Sharpe: %.4f  (SR_per_obs=%.4f, SR_annualized=%.4f, "
            "n_trials=%d, n_obs=%d)",
            dsr, sr_per_obs, sr_annualized, effective_trials, n_obs,
        )

        # ── T4.3: FIXED PBO — genuine per-fold OOS matrix ─────────────────────
        # Build a (n_folds x n_configs) matrix by replaying WF on top trials.
        # Never tile a single scalar: tiling makes IS-best mechanically OOS-best,
        # collapsing PBO to 0.0 by construction (the false-confidence bug).
        pbo = self._compute_pbo(completed, best_params)
        if pbo is None:
            log.info(
                "PBO: UNDEFINED (too few comparable OOS trials for a genuine matrix)"
            )
        else:
            log.info("PBO: %.4f", pbo)

        # ── T4.4: Locked holdout evaluation (ONCE) ─────────────────────────────
        holdout_score   = 0.0
        holdout_metrics: Dict[str, Any] = {}
        if len(self.objective.df_holdout) > 0:
            log.info(
                "Evaluating best config on HOLDOUT (%d bars, never seen by optimizer)...",
                len(self.objective.df_holdout),
            )
            try:
                ho_signals = self.objective.strategy.generate_signals(
                    self.objective.df_holdout, best_params
                )
                ho_result = self.objective.engine.run(
                    self.objective.df_holdout, ho_signals, best_params
                )
                holdout_score = compute_score_v2(
                    ho_result.trades, ho_result.equity_curve,
                    n_trials   = effective_trials,
                    min_trades = self.objective.min_trades_eff,
                )
                holdout_metrics = ho_result.metrics
                log.info(
                    "Holdout: trades=%d  score=%.4f  PF=%.2f  WR=%.1f%%  "
                    "MDD=%.1f%%  Sharpe=%.2f",
                    len(ho_result.trades), holdout_score,
                    holdout_metrics.get("profit_factor", 0),
                    holdout_metrics.get("win_rate", 0) * 100,
                    holdout_metrics.get("max_drawdown", 0) * 100,
                    holdout_metrics.get("sharpe", 0),
                )
                # Log holdout backtest to Supabase
                rl.log_backtest(
                    symbol         = self.symbol,
                    timeframe      = self.timeframe,
                    run_uuid       = run_uuid,
                    strategy_name  = self.strategy_name,
                    is_holdout     = True,
                    params         = best_params,
                    metrics        = {
                        **holdout_metrics,
                        "deflated_sharpe":  dsr,
                        "pbo":              pbo,
                        "effective_trials": effective_trials,
                        "holdout_score":    holdout_score,
                    },
                )
            except Exception as exc:
                log.error("Holdout evaluation failed: %s", exc)
        else:
            log.warning("Holdout set is empty; skipping holdout evaluation.")

        # ── Supabase optimization log ─────────────────────────────────────────
        rl.log_optimization(
            symbol           = self.symbol,
            timeframe        = self.timeframe,
            strategy_name    = self.strategy_name,
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

        # ── Save JSON output ──────────────────────────────────────────────────
        is_robust = wf_result.is_robust and dsr >= 0.5 and holdout_score > 0.0
        output = self._build_output(
            best_params      = best_params,
            best_score       = float(best.value) if best.value is not None else 0.0,
            train_metrics    = train_result.metrics,
            holdout_score    = holdout_score,
            holdout_metrics  = holdout_metrics,
            dsr              = dsr,
            pbo              = pbo,
            effective_trials = effective_trials,
            wf_result        = wf_result,
            is_robust        = is_robust,
            trades_per_year  = trades_per_year,
        )
        out_path = OUTPUT_DIR / f"strategy_best_{self.strategy_name}.json"
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        log.info("Saved %s", out_path)

        return output

    # ── PBO from genuine per-fold OOS matrix ──────────────────────────────────

    def _compute_pbo(
        self,
        completed,
        best_params: Dict[str, Any],
        *,
        max_configs: int = 24,
        min_configs: int = 4,
    ) -> Optional[float]:
        """
        Build a genuine (n_folds x n_configs) per-period OOS matrix and
        compute PBO from it.

        For each of up to max_configs top completed trials we replay full
        walk-forward (train data only) and collect the per-fold OOS scores.
        Trials are kept only if they yield the same number of OOS folds as the
        majority, ensuring the matrix is rectangular and rows are aligned to
        the same time blocks.

        Returns float in [0, 1] or None when too few comparable trials exist.
        """
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
            n_splits = OPTIMIZER["walk_forward_splits"],
        )

        columns: list = []
        for t in scored:
            params = t.user_attrs.get("full_params") or t.params
            try:
                res = wf.analyze(
                    self.objective.df,
                    self.objective.strategy.generate_signals,
                    params,
                )
            except Exception as exc:
                log.debug("PBO: WF replay failed for trial #%d: %s", t.number, exc)
                continue
            fold_scores = [f.oos_score for f in res.folds]
            if fold_scores:
                columns.append(fold_scores)

        if len(columns) < min_configs:
            return None

        # Align to the most common fold count so the matrix is rectangular
        n_folds = Counter(len(c) for c in columns).most_common(1)[0][0]
        aligned = [c for c in columns if len(c) == n_folds]
        if len(aligned) < min_configs:
            return None

        # perf_matrix: rows = OOS folds (genuine OOS observations), cols = configs
        perf_matrix = np.array(aligned, dtype=float).T   # (n_folds, n_configs)
        return pbo_from_oos_matrix(perf_matrix)

    # ── Study factory ──────────────────────────────────────────────────────────

    def _create_study(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials          = min(50, max(10, self.n_trials // 4)),
            multivariate              = True,
            warn_independent_sampling = False,
            seed                      = 42,
        )
        pruner = optuna.pruners.HyperbandPruner(
            min_resource     = 3,
            max_resource     = 200,
            reduction_factor = 3,
        )
        study_name = (
            f"strategy_opt_{self.strategy_name}_{self.symbol}_{self.timeframe}"
        )
        study = optuna.create_study(
            study_name     = study_name,
            storage        = OPTIMIZER["storage"],
            direction      = "maximize",
            sampler        = sampler,
            pruner         = pruner,
            load_if_exists = True,
        )
        n_existing = len(study.trials)
        if n_existing > 0:
            log.info("Resuming study '%s' with %d existing trials.", study_name, n_existing)
        else:
            log.info("Starting fresh study '%s'.", study_name)
        return study

    # ── Trial callback ─────────────────────────────────────────────────────────

    @staticmethod
    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n > 0 and n % 50 == 0:
            log.info(
                "Progress: %4d trials | Best=%.6f (trial #%d)",
                n, study.best_value, study.best_trial.number,
            )

    # ── Output builder ─────────────────────────────────────────────────────────

    def _build_output(
        self,
        best_params:      Dict[str, Any],
        best_score:       float,
        train_metrics:    Dict[str, Any],
        holdout_score:    float,
        holdout_metrics:  Dict[str, Any],
        dsr:              float,
        pbo:              Optional[float],
        effective_trials: int,
        wf_result,
        is_robust:        bool,
        trades_per_year:  float,
    ) -> Dict[str, Any]:
        def _clean(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        return {
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "strategy":         self.strategy_name,
            "symbol":           self.symbol,
            "timeframe":        self.timeframe,
            "is_robust":        is_robust,
            "best_score":       round(best_score, 6),      # train OOS (Optuna best)
            "holdout_score":    round(holdout_score, 6),   # once-only locked holdout
            "deflated_sharpe":  round(dsr, 6),
            "pbo":              (round(pbo, 6) if pbo is not None else None),
            "effective_trials": int(effective_trials),
            "trades_per_year":      round(trades_per_year, 2),  # annualized; basis BARS_PER_YEAR
            "min_trades_per_year":  self.min_trades_per_year,   # frequency floor applied in objective
            "walk_forward": {
                "is_robust":       wf_result.is_robust,
                "pass_rate":       round(wf_result.pass_rate, 3),
                "avg_oos_score":   round(wf_result.avg_oos_score, 6),
                "avg_degradation": round(wf_result.avg_degradation, 3),
            },
            "holdout_metrics": {
                k: _clean(v) for k, v in holdout_metrics.items()
            },
            "train_metrics": {
                k: _clean(v) for k, v in train_metrics.items()
            },
            "best_params": {
                k: _clean(v) for k, v in best_params.items()
            },
        }


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="APEX Single-Strategy Optimizer")
    parser.add_argument("--strategy",  required=True,
                        help="Strategy name (e.g. short_term_reversal)")
    parser.add_argument("--symbol",    default="MES")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--trials",    type=int,   default=None,
                        help="Number of Optuna trials (default: data-derived recommendation)")
    parser.add_argument("--jobs",      type=int,   default=1)
    parser.add_argument("--timeout",   type=float, default=None,
                        help="Timeout in hours (0 = none)")
    parser.add_argument("--no-wf",     action="store_true",
                        help="Skip walk-forward validation (faster, less robust)")
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--min-trades-per-year", type=float, default=50.0,
                        help="Minimum annualized trade frequency (default 50). "
                             "Configs trading below this floor have their objective "
                             "score linearly penalized toward 0.")
    parser.add_argument("--days", type=int, default=None,
                        help="History window in days (default: timeframe max)")
    parser.add_argument("--refetch", action="store_true",
                        help="Force a fresh Databento fetch instead of using cache")
    args = parser.parse_args()

    optimizer = StrategyOptimizer(
        strategy_name       = args.strategy,
        symbol              = args.symbol,
        timeframe           = args.timeframe,
        n_trials            = args.trials,
        n_jobs              = args.jobs,
        timeout_h           = args.timeout,
        holdout_frac        = args.holdout_frac,
        use_wf              = not args.no_wf,
        min_trades_per_year = args.min_trades_per_year,
        days                = args.days,
        use_cache           = not args.refetch,
    )
    result = optimizer.run()

    # ── Headline verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STRATEGY OPTIMIZER RESULT: {args.strategy}")
    print("=" * 60)
    print(f"  Best train OOS score : {result.get('best_score', 'N/A'):.4f}")
    print(f"  Holdout score        : {result.get('holdout_score', 'N/A'):.4f}")
    print(f"  Deflated Sharpe (DSR): {result.get('deflated_sharpe', 'N/A'):.4f}")
    print(f"  Trades / year        : {result.get('trades_per_year', 0):.1f}  "
          f"(floor={result.get('min_trades_per_year', 0):.0f})")
    pbo_val = result.get("pbo")
    print(f"  PBO                  : {'UNDEFINED' if pbo_val is None else f'{pbo_val:.4f}'}")
    wf = result.get("walk_forward", {})
    print(f"  WF robust            : {wf.get('is_robust', 'N/A')}  "
          f"(pass_rate={wf.get('pass_rate', 0):.0%})")
    hm = result.get("holdout_metrics", {})
    print(f"  Holdout PF           : {hm.get('profit_factor', 0):.2f}")
    print(f"  Holdout Sharpe       : {hm.get('sharpe', 0):.2f}")
    print(f"  Holdout Trades       : {hm.get('total_trades', 0)}")
    print(f"  Holdout MaxDD        : {hm.get('max_drawdown', 0):.1%}")
    print(f"  Holdout net PnL      : {hm.get('gross_profit', 0) - hm.get('gross_loss', 0):.0f}")
    print(f"  Is robust            : {result.get('is_robust', False)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
