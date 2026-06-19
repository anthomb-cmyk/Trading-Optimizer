-- APEX Optimizer - honest-engine logging schema (additive, idempotent)
--
-- Purpose: log EVERY pipeline outcome and important information, not just trades.
--   runs            one row per top-level invocation (backtest/optimize/fetch/live/test)
--   events          structured log stream (phases, validation, risk halts, notes)
--   backtest_runs   one row per backtest evaluation, including honesty metrics
--   data_fetches    one row per data pull (provenance + integrity)
--   optimization_runs is extended with honesty + provenance columns
--
-- Auth note: the backend should write with the Supabase service_role key
-- (kept in .env, never committed). The anon key also works while RLS is
-- disabled (the current default for these raw-SQL tables), matching the
-- existing `trades` logging. run_uuid columns are plain (no FK) so best-effort
-- logging never fails on a missing parent row during a partial outage.

-- ── Spine: one row per top-level pipeline invocation ────────────────────────
CREATE TABLE IF NOT EXISTS runs (
    run_uuid     UUID PRIMARY KEY,
    kind         TEXT NOT NULL,                    -- 'backtest'|'optimize'|'fetch'|'live'|'walk_forward'|'test'
    status       TEXT NOT NULL DEFAULT 'running',  -- 'running'|'ok'|'error'
    symbol       TEXT,
    timeframe    TEXT,
    git_sha      TEXT,
    config       JSONB,
    summary      JSONB,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS runs_kind_idx    ON runs (kind);
CREATE INDEX IF NOT EXISTS runs_started_idx ON runs (started_at DESC);

-- ── General structured event / log stream ───────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    id         BIGSERIAL PRIMARY KEY,
    run_uuid   UUID,
    ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level      TEXT NOT NULL DEFAULT 'INFO',       -- 'INFO'|'WARN'|'ERROR'
    kind       TEXT,                               -- 'phase'|'validation'|'risk_halt'|'fill'|'data'|'note'
    message    TEXT,
    payload    JSONB
);
CREATE INDEX IF NOT EXISTS events_run_idx  ON events (run_uuid);
CREATE INDEX IF NOT EXISTS events_kind_idx ON events (kind);
CREATE INDEX IF NOT EXISTS events_ts_idx   ON events (ts DESC);

-- ── One row per backtest evaluation (incl. honesty metrics) ─────────────────
CREATE TABLE IF NOT EXISTS backtest_runs (
    id               BIGSERIAL PRIMARY KEY,
    run_uuid         UUID,
    symbol           TEXT NOT NULL,
    timeframe        TEXT NOT NULL,
    strategy_name    TEXT,
    data_source      TEXT,                         -- 'databento'|'yfinance'|...
    bar_start        TIMESTAMPTZ,
    bar_end          TIMESTAMPTZ,
    n_bars           INTEGER,
    is_holdout       BOOLEAN NOT NULL DEFAULT FALSE,
    params           JSONB,
    costs            JSONB,                         -- commission/slippage assumptions used
    total_trades     INTEGER,
    win_rate         NUMERIC(6,4),
    profit_factor    NUMERIC(8,4),
    max_drawdown     NUMERIC(8,4),
    sharpe           NUMERIC(8,4),
    deflated_sharpe  NUMERIC(8,4),                  -- honesty metric (Bailey/Lopez de Prado)
    pbo              NUMERIC(6,4),                  -- probability of backtest overfitting
    effective_trials INTEGER,
    net_pnl          NUMERIC(14,2),
    metrics          JSONB,                         -- full metrics blob
    git_sha          TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS bt_runs_symbol_idx   ON backtest_runs (symbol);
CREATE INDEX IF NOT EXISTS bt_runs_strategy_idx ON backtest_runs (strategy_name);
CREATE INDEX IF NOT EXISTS bt_runs_created_idx  ON backtest_runs (created_at DESC);

-- ── One row per data fetch (provenance + integrity) ─────────────────────────
CREATE TABLE IF NOT EXISTS data_fetches (
    id           BIGSERIAL PRIMARY KEY,
    run_uuid     UUID,
    source       TEXT NOT NULL,                     -- 'databento'|'yfinance'|...
    symbol       TEXT NOT NULL,
    timeframe    TEXT,
    bar_start    TIMESTAMPTZ,
    bar_end      TIMESTAMPTZ,
    n_bars       INTEGER,
    gaps_found   INTEGER,
    cost_units   NUMERIC(12,4),                     -- credits/bytes/USD where known
    ok           BOOLEAN NOT NULL DEFAULT TRUE,
    message      TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS data_fetches_symbol_idx  ON data_fetches (symbol);
CREATE INDEX IF NOT EXISTS data_fetches_created_idx ON data_fetches (created_at DESC);

-- ── Extend optimization_runs with honesty + provenance columns ──────────────
ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS run_uuid         UUID;
ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS deflated_sharpe  NUMERIC(8,4);
ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS pbo              NUMERIC(6,4);
ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS effective_trials INTEGER;
ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS oos_score        NUMERIC(10,6);
ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS holdout_score    NUMERIC(10,6);
ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS data_source      TEXT;
ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS git_sha          TEXT;
ALTER TABLE optimization_runs ADD COLUMN IF NOT EXISTS status           TEXT;
