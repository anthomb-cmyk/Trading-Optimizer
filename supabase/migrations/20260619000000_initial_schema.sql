-- APEX Optimizer — initial schema

CREATE TABLE IF NOT EXISTS trades (
    id               BIGSERIAL PRIMARY KEY,
    date             DATE          NOT NULL,
    symbol           TEXT          NOT NULL,
    direction        TEXT          NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    entry_time       TIMESTAMPTZ   NOT NULL,
    entry_price      NUMERIC(12,4) NOT NULL,
    exit_time        TIMESTAMPTZ   NOT NULL,
    exit_price       NUMERIC(12,4) NOT NULL,
    quantity         INTEGER       NOT NULL,
    gross_pnl        NUMERIC(12,2) NOT NULL,
    strategy_name    TEXT,
    confluence_score INTEGER,
    stop_loss        NUMERIC(12,4),
    take_profit      NUMERIC(12,4),
    exit_reason      TEXT,
    confluences_used TEXT,
    bias_score       INTEGER,
    kill_zone        BOOLEAN       NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS trades_date_idx    ON trades (date);
CREATE INDEX IF NOT EXISTS trades_symbol_idx  ON trades (symbol);
CREATE INDEX IF NOT EXISTS trades_strategy_idx ON trades (strategy_name);

CREATE TABLE IF NOT EXISTS optimization_runs (
    id             BIGSERIAL PRIMARY KEY,
    symbol         TEXT          NOT NULL,
    timeframe      TEXT          NOT NULL,
    strategy_name  TEXT          NOT NULL,
    n_trials       INTEGER,
    best_score     NUMERIC(10,6),
    best_params    JSONB,
    win_rate       NUMERIC(6,4),
    profit_factor  NUMERIC(8,4),
    max_drawdown   NUMERIC(6,4),
    total_trades   INTEGER,
    run_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS opt_runs_symbol_idx   ON optimization_runs (symbol);
CREATE INDEX IF NOT EXISTS opt_runs_strategy_idx ON optimization_runs (strategy_name);
CREATE INDEX IF NOT EXISTS opt_runs_run_at_idx   ON optimization_runs (run_at DESC);
