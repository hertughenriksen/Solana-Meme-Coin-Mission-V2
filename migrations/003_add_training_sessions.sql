-- Migration 003: training_sessions table
CREATE TABLE IF NOT EXISTS training_sessions (
    id          SERIAL      PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ,
    note        TEXT
);

INSERT INTO training_sessions (started_at)
SELECT NOW()
WHERE NOT EXISTS (SELECT 1 FROM training_sessions LIMIT 1);

CREATE OR REPLACE VIEW training_progress AS
SELECT
    MIN(ts.started_at)                                        AS collection_started_at,
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
