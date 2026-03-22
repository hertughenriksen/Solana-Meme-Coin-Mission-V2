-- Migration 002: add source_wallet to signals so copy-trade stats can be attributed
-- to the wallet that triggered the signal.
ALTER TABLE signals ADD COLUMN IF NOT EXISTS source_wallet VARCHAR(44);

CREATE INDEX IF NOT EXISTS idx_signals_source_wallet
    ON signals(source_wallet)
    WHERE source_wallet IS NOT NULL;
