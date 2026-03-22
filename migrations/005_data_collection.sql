-- Migration 005: data-collection build
-- Ensures the schema is ready for the free-API poller and price-candle writes.

-- Make sure the unique constraint on price_candles_2s exists so ON CONFLICT works.
-- (The original schema uses a bigserial PK but no unique index on (mint, ts);
--  add one so duplicate 2s candles from the price feed are silently ignored.)
CREATE UNIQUE INDEX IF NOT EXISTS idx_price_candles_2s_mint_ts_unique
    ON price_candles_2s (mint, ts);

-- Ensure all detection-time columns exist (idempotent — same as 004 but safe to re-run).
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

ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS source_wallet TEXT;

ALTER TABLE deployer_wallets
    ADD COLUMN IF NOT EXISTS win_rate DOUBLE PRECISION DEFAULT 0.5;

CREATE TABLE IF NOT EXISTS autoresearch_runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    total_experiments INTEGER DEFAULT 0,
    best_kelly_return DOUBLE PRECISION,
    best_params     JSONB,
    history_path    TEXT
);

-- Speed up the outcome_tracker query that fetches candles per mint.
CREATE INDEX IF NOT EXISTS idx_price_candles_2s_mint_ts_brin
    ON price_candles_2s USING brin (mint, ts);

-- Speed up write_token_detection upserts.
CREATE INDEX IF NOT EXISTS idx_tokens_price_usd
    ON tokens (price_usd_at_detection)
    WHERE price_usd_at_detection IS NOT NULL AND price_usd_at_detection > 0;
