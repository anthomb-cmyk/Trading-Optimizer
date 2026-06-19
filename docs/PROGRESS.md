# Build Progress - APEX Trading Optimizer

Phase 1 = the honest measurement system. Full spec: [PHASE1_SPEC.md](PHASE1_SPEC.md).

## 2026-06-19

### Supabase logging foundation - DONE (not yet deployed)
- `supabase/migrations/20260619010000_honest_engine_logging.sql`: new tables `runs`, `events`, `backtest_runs`, `data_fetches`, plus honesty/provenance columns on `optimization_runs` (`deflated_sharpe`, `pbo`, `effective_trials`, `oos_score`, `holdout_score`, ...). Additive + idempotent.
- `config/run_logger.py`: logging backbone (`start_run`, `finish_run`, `log_event`, `log_backtest`, `log_optimization`, `log_data_fetch`). Never raises, no-ops without env, mirrors to file log, sanitizes numpy/pandas/datetime.
- `docs/SUPABASE_LOGGING.md`: schema, deploy steps, auth notes.
- Deploy still pending: push to `main` (GitHub Action) or `supabase db push`. The chat MCP token has no access to project `umqdxhvilenqmrbqawhi`, so deploy happens via the repo, not from chat.

### T1.1 causal swing detection - DONE, verified
- `data/swings.py` (new, numpy-only): confirm-and-shift detection. Pivot at `i` is flagged only at confirmation bar `i+lb`, so no flag depends on future bars.
- `data/loader.py`: keeps `swing_high`/`swing_low` booleans, adds causal `swing_high_price`/`swing_low_price` (= `high/low.shift(lb).where(flag)`, the true pivot price with no look-ahead).
- 6 consumers repointed to the price columns: `base_strategy`, `daily_bias_intraday`, `ote_fibonacci`, `market_structure_shift`, `stop_hunt_continuation`, `liquidity_sweep`.
- `tests/test_causality.py`: 3 tests (no-repaint, pure causality, swing-price correctness) pass on Python 3.13.

## Open items / next

**Inputs needed from Anthony**
- Databento API key (free, $125 credits) for the real-data pipeline (T3.1).
- Real account size + per-trade risk % to replace the $1M / $10k placeholders (T6.1).

**Decisions**
- Commit + push now (triggers the Supabase auto-deploy Action), or hold until more of Phase 1 lands?
- Confirm `SUPABASE_KEY` in `.env` is the service_role key (recommended for a server-side bot).

**Environment**
- Need a Python 3.11 venv to run the full pipeline (vectorbt/numba break on 3.12+). System Python is 3.13; the numpy-only causality tests run there, but backtests do not.

**Next tickets (no external input needed)**
- T1.2: detector causality - `detect_order_blocks` uses `shift(-j)` (forward), plus `detect_fvg` / `detect_bos` / `detect_choch` need the same no-repaint guarantee as swings.
- T1.3: per-fold signal generation (stop generating signals on the full series before the walk-forward split).
- T2.x: realistic cost/fill model. T4.2: Deflated Sharpe + PBO metrics (feed the new Supabase columns).

## 2026-06-19 - Wave 1 complete (data pipeline + look-ahead elimination)

### T3 Databento pipeline - DONE
- `data/databento_loader.py` (new): GLBX.MDP3 fetch, continuous `ES.c.0`/`NQ.c.0`/`GC.c.0` (calendar roll, back-adjusted), 1m native + resample to 5m/15m, parquet cache, integrity checks, logs via `run_logger.log_data_fetch`.
- `data/loader.py`: `load_bars(..., source=)` auto-selects Databento for futures; yfinance only as explicit opt-in; SPY/QQQ/GLD proxy off the default path. `_add_indicators` and swings untouched.
- Validated on real ES/MES 15m data; smoke test runs all 10 strategies with no errors.

### T1.2 detector causality - DONE
- `strategies/detectors.py` (new): pure causal detectors. `detect_order_blocks` flag moved to the displacement bar and now returns the OB candle's zone (`_high`/`_low`) carried causally to that bar. FVG/BOS/CHoCH verified causal.
- `strategies/base_strategy.py`: detectors delegated to `detectors.py`; 5 callers updated.
- `tests/test_detector_causality.py`: 7 tests incl. OB-zone correctness + no-repaint.

### T1.3 per-fold signals - DONE
- `backtesting/walk_forward.py` + `optimizer/sunday_optimizer.py`: signals generated per fold with a causal IS-side warmup prefix; OOS append-invariance proven.
- `tests/test_walkforward_causality.py`: 5 tests.

Look-ahead elimination (Epic 1) is complete: swings, detectors, and walk-forward are all causal and regression-tested. All three causality suites pass together.

### Next: Wave 2
- T2.x realistic cost/fill model (engine.py); T4.1/4.2/4.3/4.4 purged CV + Deflated Sharpe + PBO + trial budget + locked holdout; T4.5 scoring off win-rate; T5.1/T5.2 buy-and-hold + noise honesty tests; T6.1 config realism (account size/risk + flip DEFAULT_INSTRUMENT to MES + CME-aware gap detection).
- Still needs from Anthony: account size + per-trade risk %.
- Known Phase 2 strategy-logic bugs confirmed live: fvg_retest duplicate-confluence over-firing; asian_session breakout precedence (never fires).

## 2026-06-19 - Wave 2 complete (costs, validation, honesty metrics, config)

### Engine costs/fills (T2.x, T5.1) - DONE
- `backtesting/engine.py`: two-sided regime-aware slippage, round-trip commission, pessimistic fills (gap-through stops fill at open; TP needs trade-through; SL assumed before TP intrabar), mark-to-market equity. `tests/test_engine_costs.py` (5 tests).

### Honesty metrics + scoring (T4.2, T4.5) - DONE
- `backtesting/metrics.py`: `deflated_sharpe_ratio`, `probability_backtest_overfitting` (CSCV), `minimum_track_record_length`, `compute_score_v2` (demotes win_rate). `tests/test_metrics.py` (13 tests).

### Validation pipeline (T4.1, T4.3, T4.4, T5.2) - DONE
- `backtesting/walk_forward.py`: purge+embargo splits, CPCV, `recommended_max_trials`.
- `optimizer/sunday_optimizer.py`: locked 20% holdout, data-derived trial budget (~5,113 not 500k), `compute_score_v2` objective, DSR/PBO computed and logged to Supabase. `tests/test_validation.py` (8 tests; noise gate holdout DSR=0.0).

### Config realism (T6.1) - DONE
- `config/settings.py`: account 50k (placeholder), risk 500/trade, `MAX_CONTRACTS_PER_TRADE=10`, `DEFAULT_INSTRUMENT`/`TEST_SYMBOL` -> MES.

### First honest backtest (real MES, 7738 bars, ~4 months, unoptimized defaults, realistic costs)
- order_block PF 0.80 | fvg_retest PF 0.65 | liquidity_sweep PF 0.18 | market_structure PF 1.00 | kill_zone PF 1.70 (25 trades, NOT yet validated). Most ICT strategies lose with causal signals + real costs, as expected.

## Phase 1 status: COMPLETE (Epics 1-6 + Supabase logging). 28 tests across 6 suites green.

## Phase 2 plan
Trustworthiness fixes first (engine must be believable before validating strategies):
- Cache ignores `days` (stale smaller cache returned) -> fix cache keying in loader.
- CME-aware gap detection (false positives on weekend/session breaks) in databento_loader.
- Sharpe inconsistency (fvg_retest PF 0.65 but Sharpe 4.6) in metrics -> verify/fix (DSR depends on it).
Then strategy logic bugs (see Appendix A): fvg duplicate confluence, asian_session precedence, ote `retracement_min` unused, market_structure rsi window, stop_hunt hardcoded 0.6, breaker displacement body filter.
Then: run the optimizer (DSR/PBO/holdout) per strategy on real data -> keep only validated survivors; pivot the strategy layer if none survive.

### Open inputs / ops
- Anthony to confirm account size (placeholder 50k) and that the Supabase migration deployed (Actions) + service_role key is in `.env`.
