CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

CREATE TABLE IF NOT EXISTS trades (
    id                    UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    mint                  VARCHAR(44) NOT NULL,
    strategy_track        VARCHAR(20) NOT NULL,
    status                VARCHAR(20) NOT NULL DEFAULT 'pending',
    entry_price_usd       DECIMAL(20,10),
    entry_amount_sol      DECIMAL(20,9)  NOT NULL,
    entry_tx              VARCHAR(88),
    entered_at            TIMESTAMPTZ,
    exit_price_usd        DECIMAL(20,10),
    exit_amount_sol       DECIMAL(20,9),
    exit_tx               VARCHAR(88),
    exited_at             TIMESTAMPTZ,
    pnl_sol               DECIMAL(20,9),
    pnl_pct               DECIMAL(10,6),
    peak_multiplier       DECIMAL(10,4),
    jito_tip_lamports     BIGINT,
    ml_tabular_score      DECIMAL(5,4),
    ml_transformer_score  DECIMAL(5,4),
    ml_gnn_score          DECIMAL(5,4),
    ml_nlp_score          DECIMAL(5,4),
    ml_ensemble_score     DECIMAL(5,4),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trades_mint       ON trades(mint);
CREATE INDEX IF NOT EXISTS idx_trades_status     ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy   ON trades(strategy_track);

CREATE TABLE IF NOT EXISTS tokens (
    mint                        VARCHAR(44) PRIMARY KEY,
    first_seen_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dex                         VARCHAR(20),
    liquidity_usd               DECIMAL(20,2),
    market_cap_usd              DECIMAL(20,2),
    deployer_wallet             VARCHAR(44),
    mint_authority_disabled     BOOLEAN,
    freeze_authority_disabled   BOOLEAN,
    lp_locked                   BOOLEAN,
    lp_lock_days                INTEGER,
    dev_holding_pct             DECIMAL(5,4),
    sniper_concentration_pct    DECIMAL(5,4),
    top_10_holder_pct           DECIMAL(5,4),
    filter_passed               BOOLEAN,
    filter_rejection_reason     TEXT,
    ml_ensemble_score           DECIMAL(5,4),
    outcome                     VARCHAR(20),
    peak_multiplier             DECIMAL(10,4),
    peak_at                     TIMESTAMPTZ,
    died_at                     TIMESTAMPTZ,
    ml_label                    SMALLINT,
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tokens_deployer   ON tokens(deployer_wallet);
CREATE INDEX IF NOT EXISTS idx_tokens_filter     ON tokens(filter_passed);
CREATE INDEX IF NOT EXISTS idx_tokens_ml_label   ON tokens(ml_label);
CREATE INDEX IF NOT EXISTS idx_tokens_first_seen ON tokens(first_seen_at DESC);

CREATE TABLE IF NOT EXISTS deployer_wallets (
    wallet              VARCHAR(44) PRIMARY KEY,
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_tokens        INTEGER     DEFAULT 0,
    rug_count           INTEGER     DEFAULT 0,
    pump_count          INTEGER     DEFAULT 0,
    rug_rate            DECIMAL(5,4),
    is_known_rugger     BOOLEAN     DEFAULT FALSE,
    last_activity       TIMESTAMPTZ,
    notes               TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS copy_wallets (
    wallet              VARCHAR(44) PRIMARY KEY,
    alias               VARCHAR(50),
    added_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active           BOOLEAN     DEFAULT TRUE,
    total_trades        INTEGER     DEFAULT 0,
    winning_trades      INTEGER     DEFAULT 0,
    win_rate            DECIMAL(5,4),
    avg_win_multiplier  DECIMAL(10,4),
    avg_hold_minutes    DECIMAL(10,2),
    total_pnl_sol       DECIMAL(20,9),
    source              VARCHAR(50),
    notes               TEXT,
    last_trade_at       TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS signals (
    id                  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    mint                VARCHAR(44) NOT NULL,
    source              VARCHAR(20) NOT NULL,
    signal_type         VARCHAR(30) NOT NULL,
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    twitter_mentions_5m  INTEGER,
    telegram_mentions_5m INTEGER,
    sentiment_score      DECIMAL(5,4),
    kol_mention          BOOLEAN DEFAULT FALSE,
    traded              BOOLEAN     DEFAULT FALSE,
    trade_id            UUID        REFERENCES trades(id),
    filter_passed       BOOLEAN,
    filter_rejection    TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_mint        ON signals(mint);
CREATE INDEX IF NOT EXISTS idx_signals_source      ON signals(source);
CREATE INDEX IF NOT EXISTS idx_signals_detected_at ON signals(detected_at DESC);

CREATE TABLE IF NOT EXISTS price_candles_2s (
    id          BIGSERIAL   PRIMARY KEY,
    mint        VARCHAR(44) NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    open        DECIMAL(20,12) NOT NULL,
    high        DECIMAL(20,12) NOT NULL,
    low         DECIMAL(20,12) NOT NULL,
    close       DECIMAL(20,12) NOT NULL,
    volume      DECIMAL(20,6)  NOT NULL,
    buy_count   INTEGER     NOT NULL DEFAULT 0,
    sell_count  INTEGER     NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_candles_mint_ts ON price_candles_2s(mint, ts DESC);

CREATE OR REPLACE VIEW session_stats AS
SELECT
    COUNT(*) AS total_trades,
    SUM(CASE WHEN pnl_sol > 0  THEN 1 ELSE 0 END) AS winning_trades,
    SUM(CASE WHEN pnl_sol <= 0 AND status='closed' THEN 1 ELSE 0 END) AS losing_trades,
    ROUND(SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END)::DECIMAL / NULLIF(COUNT(*), 0), 4) AS win_rate,
    SUM(COALESCE(pnl_sol, 0)) AS total_pnl_sol,
    MAX(COALESCE(pnl_pct, 0)) AS best_trade_pct,
    MIN(COALESCE(pnl_pct, 0)) AS worst_trade_pct,
    AVG(EXTRACT(EPOCH FROM (exited_at - entered_at)) / 60) AS avg_hold_minutes,
    SUM(COALESCE(jito_tip_lamports, 0)) / 1e9 AS jito_tips_paid_sol,
    COUNT(*) FILTER (WHERE status IN ('confirmed','partial_exit')) AS open_positions
FROM trades
WHERE created_at >= NOW() - INTERVAL '24 hours';

CREATE OR REPLACE FUNCTION update_deployer_stats() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.outcome IS NOT NULL AND OLD.outcome IS NULL AND NEW.deployer_wallet IS NOT NULL THEN
        INSERT INTO deployer_wallets (wallet, first_seen_at) VALUES (NEW.deployer_wallet, NOW())
        ON CONFLICT (wallet) DO NOTHING;
        UPDATE deployer_wallets SET
            total_tokens = total_tokens + 1,
            rug_count    = rug_count  + CASE WHEN NEW.outcome IN ('rug','honeypot') THEN 1 ELSE 0 END,
            pump_count   = pump_count + CASE WHEN NEW.outcome = 'pump'             THEN 1 ELSE 0 END,
            rug_rate     = (rug_count + CASE WHEN NEW.outcome IN ('rug','honeypot') THEN 1 ELSE 0 END)::DECIMAL / NULLIF(total_tokens + 1, 0),
            last_activity = NOW(), updated_at = NOW()
        WHERE wallet = NEW.deployer_wallet;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_token_outcome ON tokens;
CREATE TRIGGER trg_token_outcome AFTER UPDATE ON tokens FOR EACH ROW EXECUTE FUNCTION update_deployer_stats();

CREATE INDEX IF NOT EXISTS idx_tokens_outcome_label ON tokens(outcome, ml_label) WHERE ml_label IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tokens_deployer_outcome ON tokens(deployer_wallet, outcome) WHERE deployer_wallet IS NOT NULL;
