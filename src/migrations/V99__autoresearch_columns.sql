-- V99__autoresearch_columns.sql
-- Adds columns that the autoresearch experiment runner and the fixed
-- get_training_stats() Rust function expect to be present.

-- Tokens: columns populated by the data collector and outcome_tracker.
ALTER TABLE tokens
    ADD COLUMN IF NOT EXISTS liquidity_usd_at_detection    DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS market_cap_usd_at_detection   DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sniper_concentration_pct      DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dev_holding_pct               DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS deployer_rug_rate             DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS lp_lock_days                  INTEGER,
    ADD COLUMN IF NOT EXISTS buy_sell_ratio_at_detection   DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_usd_at_detection        DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS top_holders_json              JSONB;

-- Signals: source column needed by the fixed get_training_stats() query.
ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS source_wallet TEXT;

-- Deployer wallets: win_rate column.
ALTER TABLE deployer_wallets
    ADD COLUMN IF NOT EXISTS win_rate DOUBLE PRECISION DEFAULT 0.5;

-- Autoresearch experiment runs (audit log for CI/CD).
CREATE TABLE IF NOT EXISTS autoresearch_runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    total_experiments INTEGER DEFAULT 0,
    best_kelly_return DOUBLE PRECISION,
    best_params     JSONB,
    history_path    TEXT
);

-- Index to speed up the per-mint candle query in experiment_runner.py.
CREATE INDEX IF NOT EXISTS idx_price_candles_2s_mint_ts
    ON price_candles_2s (mint, ts);
