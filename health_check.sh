#!/usr/bin/env bash
# health_check.sh — run this any time to verify data collection is working.
# Usage: bash health_check.sh
# Requires: psql, DATABASE_URL in environment or .env file

set -euo pipefail

if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

if [ -z "${DATABASE_URL:-}" ]; then
    echo "ERROR: DATABASE_URL not set. Add it to .env or export it."
    exit 1
fi

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo ""
echo "=== DATA COLLECTION HEALTH CHECK ==="
echo ""

psql "$DATABASE_URL" -x -c "
SELECT
    COUNT(*)                                                         AS total_tokens,
    COUNT(*) FILTER (WHERE price_usd_at_detection > 0)              AS has_price,
    COUNT(*) FILTER (WHERE ml_label IS NOT NULL)                     AS labeled,
    COUNT(*) FILTER (WHERE ml_label = 1)                             AS wins,
    COUNT(*) FILTER (WHERE ml_label = 0)                             AS losses,
    ROUND(
        COUNT(*) FILTER (WHERE ml_label = 1)::numeric /
        NULLIF(COUNT(*) FILTER (WHERE ml_label IS NOT NULL), 0) * 100, 1
    )                                                                AS win_pct,
    ROUND(
        EXTRACT(EPOCH FROM (MAX(first_seen_at) - MIN(first_seen_at))) / 3600.0
    , 1)                                                             AS hours_collected,
    MAX(first_seen_at)                                               AS latest_token
FROM tokens;
" 2>/dev/null

echo ""
echo "--- Candle data ---"
psql "$DATABASE_URL" -x -c "
SELECT
    COUNT(DISTINCT mint)    AS mints_with_candles,
    COUNT(*)                AS total_candle_rows,
    MIN(ts)                 AS oldest_candle,
    MAX(ts)                 AS newest_candle
FROM price_candles_2s;
" 2>/dev/null

echo ""
echo "--- Signals ---"
psql "$DATABASE_URL" -x -c "
SELECT source, COUNT(*) AS count
FROM signals
GROUP BY source
ORDER BY count DESC;
" 2>/dev/null

echo ""
echo "--- Top filter rejections ---"
psql "$DATABASE_URL" -c "
SELECT filter_rejection_reason, COUNT(*) AS n
FROM tokens
WHERE filter_rejection_reason IS NOT NULL
GROUP BY filter_rejection_reason
ORDER BY n DESC
LIMIT 5;
" 2>/dev/null

echo ""

# Diagnose common problems
TOTAL=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM tokens;" 2>/dev/null | tr -d ' ')
HAS_PRICE=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM tokens WHERE price_usd_at_detection > 0;" 2>/dev/null | tr -d ' ')
CANDLES=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM price_candles_2s;" 2>/dev/null | tr -d ' ')
LABELED=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM tokens WHERE ml_label IS NOT NULL;" 2>/dev/null | tr -d ' ')

if [ "${TOTAL:-0}" -eq 0 ]; then
    echo -e "${RED}PROBLEM: No tokens in DB. Is the bot running? Check 'cargo run --release --bin bot'.${NC}"
elif [ "${HAS_PRICE:-0}" -eq 0 ]; then
    echo -e "${RED}PROBLEM: Tokens found but none have price_usd_at_detection set.${NC}"
    echo -e "${RED}         Check enrichment/mod.rs is being compiled and called.${NC}"
elif [ "${CANDLES:-0}" -eq 0 ]; then
    echo -e "${YELLOW}WARNING: No price candles yet. Price feed writes candles for tracked tokens.${NC}"
    echo -e "${YELLOW}         This is normal in the first few minutes.${NC}"
elif [ "${LABELED:-0}" -eq 0 ]; then
    echo -e "${YELLOW}WARNING: No labeled tokens yet.${NC}"
    echo -e "${YELLOW}         Run: python ml/scripts/outcome_tracker.py --db-url \"\$DATABASE_URL\" --loop${NC}"
else
    echo -e "${GREEN}OK: Data collection is healthy.${NC}"
    echo -e "${GREEN}   Tokens: ${TOTAL}  With price: ${HAS_PRICE}  Candles: ${CANDLES}  Labeled: ${LABELED}${NC}"
fi
echo ""
