-- Migration 003: training_sessions table
-- Records when dry-run / training mode began so the dashboard can show
-- progress toward the 2-week data-collection target.

CREATE TABLE IF NOT EXISTS training_sessions (
    id          SERIAL      PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ,
    note        TEXT
);

-- Seed with the current timestamp if no session exists yet.
INSERT INTO training_sessions (started_at)
SELECT NOW()
WHERE NOT EXISTS (SELECT 1 FROM training_sessions LIMIT 1);

-- Convenience view used by the training dashboard.
CREATE OR REPLACE VIEW training_progress AS
SELECT
    -- oldest session start = when data collection began
    MIN(ts.started_at)                                        AS collection_started_at,

    -- how many hours of price-candle / token data we have
    COALESCE(
        EXTRACT(EPOCH FROM (MAX(t.first_seen_at) - MIN(t.first_seen_at))) / 3600.0,
        0.0
    )                                                         AS hours_of_data,

    COUNT(DISTINCT t.mint)                                    AS total_tokens,
    COUNT(DISTINCT t.mint) FILTER (WHERE t.ml_label IS NOT NULL)     AS labeled_tokens,
    COUNT(DISTINCT t.mint) FILTER (WHERE t.ml_label = 1)             AS positive_labels,
    COUNT(DISTINCT t.mint) FILTER (WHERE t.ml_label = 0)             AS negative_labels,
    COUNT(DISTINCT t.mint) FILTER (WHERE t.filter_passed = true)     AS tokens_passed_filter,

    (SELECT COUNT(*) FROM signals)                            AS total_signals,
    (SELECT COUNT(*) FROM signals WHERE source = 'twitter')  AS twitter_signals,
    (SELECT COUNT(*) FROM signals WHERE source = 'telegram') AS telegram_signals,
    (SELECT COUNT(*) FROM signals WHERE source = 'yellowstone') AS yellowstone_signals,
    (SELECT COUNT(*) FROM signals WHERE source = 'copy_trade')  AS copy_trade_signals

FROM training_sessions ts
CROSS JOIN tokens t;
