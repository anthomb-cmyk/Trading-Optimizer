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
