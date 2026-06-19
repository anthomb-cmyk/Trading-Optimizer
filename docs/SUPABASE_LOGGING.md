# Supabase Logging

Goal: every outcome and important piece of information from the pipeline is logged to Supabase, not just closed trades.

## Project

- Project ref: `umqdxhvilenqmrbqawhi`
- URL: https://umqdxhvilenqmrbqawhi.supabase.co
- The backend reads `SUPABASE_URL` and `SUPABASE_KEY` from `.env` (see `.env.example`). Logging is a graceful no-op if they are unset.

## Tables

| Table | One row per | Key fields |
|---|---|---|
| `runs` | top-level invocation | `run_uuid`, `kind`, `status`, `config`, `summary`, `started_at`, `finished_at` |
| `events` | important event | `run_uuid`, `level`, `kind`, `message`, `payload` |
| `backtest_runs` | backtest evaluation | metrics incl. `deflated_sharpe`, `pbo`, `effective_trials`, `is_holdout` |
| `optimization_runs` | Optuna study | `best_params`, `best_score`, plus new `deflated_sharpe`, `pbo`, `oos_score`, `holdout_score` |
| `data_fetches` | data pull | `source`, `symbol`, `n_bars`, `gaps_found`, `cost_units` |
| `trades` | closed trade leg | (unchanged, written by `trade_logger.py`) |

Defined in:
- `supabase/migrations/20260619000000_initial_schema.sql` (trades + optimization_runs)
- `supabase/migrations/20260619010000_honest_engine_logging.sql` (runs, events, backtest_runs, data_fetches, optimization_runs extensions)

The honesty metrics (`deflated_sharpe`, `pbo`, `effective_trials`, `is_holdout`) are the point: they let us tell a real edge from a lucky backtest at a glance, straight from the database.

## How to log from code

```python
from config import run_logger as rl

run = rl.start_run("backtest", symbol="MES", timeframe="15m", config=params)
rl.log_event("Loaded 52k Databento bars (0 gaps)", run_uuid=run, kind="data")
rl.log_backtest(symbol="MES", timeframe="15m", run_uuid=run,
                strategy_name="order_block", data_source="databento",
                is_holdout=False, params=params, costs=costs, metrics=metrics)
rl.finish_run(run, status="ok", summary={"net_pnl": metrics["net_pnl"]})
```

`run_logger` never raises and sanitizes numpy/pandas/datetime values, so a logging
problem can never crash a backtest or a live trade.

## Deploying the schema

The Supabase MCP available in chat does not have access to this project, so the
schema is deployed one of two ways:

1. **GitHub Action (recommended):** push to `main`. `.github/workflows/supabase-deploy.yml`
   runs `supabase db push` using the repo secrets `SUPABASE_ACCESS_TOKEN` and
   `SUPABASE_DB_PASSWORD`. Any file under `supabase/migrations/**` triggers it.
2. **Local CLI:**
   ```bash
   supabase link --project-ref umqdxhvilenqmrbqawhi
   supabase db push
   ```

Migrations are idempotent (`CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`),
so re-running is safe.

## Auth and security

- The backend runs server-side, so use the **service_role** key as `SUPABASE_KEY`
  for reliable writes (it bypasses RLS). Keep it in `.env` only; never commit it.
- The anon key also works today because these raw-SQL tables have RLS disabled
  (matching the existing `trades` table). If a read-only dashboard is added later,
  enable RLS and add `SELECT` policies for the `anon` role.
