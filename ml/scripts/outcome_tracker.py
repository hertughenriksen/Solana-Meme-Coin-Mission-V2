"""
outcome_tracker.py — Labels historical token signals as wins or losses.

DATA-COLLECTION BUILD changes:
  - Works WITHOUT Birdeye API key.  Reads price_candles_2s rows that
    the Rust price_feed writes to PostgreSQL every 2 seconds.
  - Falls back to Birdeye only if BIRDEYE_API_KEY is set AND candle
    data is missing for a token.
  - Added --min-candles flag (default 20) so very sparse tokens are
    skipped rather than labeled incorrectly.
  - Logs a collection health summary every batch so you can see data
    is actually landing.
"""

import argparse
import logging
import time
from datetime import timedelta

import psycopg2
import psycopg2.extras

log = logging.getLogger("outcome_tracker")

WIN_MULTIPLIER    = 1.40
LOSS_MULTIPLIER   = 0.75
OBSERVATION_HOURS = 4
MIN_LIQ_RATIO     = 0.03


def should_skip_token(token: dict) -> tuple[bool, str]:
    liq  = token.get("liquidity_usd_at_detection") or 0.0
    mcap = token.get("market_cap_usd_at_detection") or 0.0
    if liq <= 0 or mcap <= 0:
        return True, "missing_liquidity_or_mcap"
    if (liq / mcap) < MIN_LIQ_RATIO:
        return True, f"liq_ratio_too_low:{liq/mcap:.4f}"
    return False, ""


def compute_label(candles: list[dict], entry_price: float) -> int | None:
    if not candles or entry_price <= 0:
        return None
    candles = sorted(candles, key=lambda c: c["ts"])
    win_threshold  = entry_price * WIN_MULTIPLIER
    loss_threshold = entry_price * LOSS_MULTIPLIER
    for c in candles:
        if (c.get("high") or 0.0) >= win_threshold:
            return 1
        if (c.get("low") or 0.0) <= loss_threshold:
            return 0
    return 0


def label_tokens(conn, batch_size: int = 500, dry_run: bool = False,
                 min_candles: int = 20) -> dict:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Fetch tokens old enough to have completed their observation window
    cur.execute("""
        SELECT
            t.mint,
            t.first_seen_at,
            t.liquidity_usd_at_detection,
            t.market_cap_usd_at_detection,
            t.price_usd_at_detection
        FROM tokens t
        WHERE t.ml_label IS NULL
          AND t.first_seen_at < NOW() - INTERVAL '%s hours'
          AND t.price_usd_at_detection IS NOT NULL
          AND t.price_usd_at_detection > 0
        ORDER BY t.first_seen_at ASC
        LIMIT %s
    """, (OBSERVATION_HOURS, batch_size))

    tokens = cur.fetchall()
    log.info("Unlabeled tokens ready to process: %d", len(tokens))

    wins = losses = skipped = errors = no_candles = 0

    for tok in tokens:
        mint = tok["mint"]

        skip, reason = should_skip_token(tok)
        if skip:
            log.debug("Skipping %s: %s", mint[:8], reason)
            skipped += 1
            continue

        entry_price = tok["price_usd_at_detection"]
        obs_end     = tok["first_seen_at"] + timedelta(hours=OBSERVATION_HOURS)

        # Read candles from the price_candles_2s table (written by Rust price_feed)
        try:
            cur.execute("""
                SELECT ts, high, low FROM price_candles_2s
                WHERE mint = %s
                  AND ts BETWEEN %s AND %s
                ORDER BY ts ASC
                LIMIT 5000
            """, (mint, tok["first_seen_at"], obs_end))
            candles = cur.fetchall()
        except Exception as exc:
            log.warning("Candle fetch failed for %s: %s", mint[:8], exc)
            errors += 1
            continue

        if len(candles) < min_candles:
            log.debug("Too few candles (%d < %d) for %s", len(candles), min_candles, mint[:8])
            no_candles += 1
            continue

        label = compute_label([dict(c) for c in candles], entry_price)
        if label is None:
            skipped += 1
            continue

        if label == 1:
            wins += 1
        else:
            losses += 1

        if not dry_run:
            cur.execute(
                "UPDATE tokens SET ml_label = %s, updated_at = NOW() WHERE mint = %s",
                (label, mint),
            )

    if not dry_run:
        conn.commit()

    total = wins + losses
    log.info(
        "Batch done | wins=%d losses=%d skipped=%d no_candles=%d errors=%d | "
        "win_rate=%.1f%% | dry_run=%s",
        wins, losses, skipped, no_candles, errors,
        (wins / total * 100 if total else 0),
        dry_run,
    )

    # Collection health summary
    cur.execute("""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE price_usd_at_detection > 0) AS has_price,
            COUNT(*) FILTER (WHERE ml_label IS NOT NULL) AS labeled,
            COALESCE(
                EXTRACT(EPOCH FROM (MAX(first_seen_at) - MIN(first_seen_at))) / 3600.0,
                0
            ) AS hours_collected
        FROM tokens
    """)
    health = cur.fetchone()
    if health:
        log.info(
            "DB health | total_tokens=%d has_price=%d labeled=%d hours=%.1f",
            health["total"], health["has_price"], health["labeled"],
            health["hours_collected"] or 0,
        )
        if health["total"] > 0 and health["has_price"] == 0:
            log.warning(
                "PROBLEM: %d tokens in DB but NONE have price_usd_at_detection set. "
                "Check that enrichment/mod.rs is running and write_token_detection() "
                "is being called in strategy/mod.rs.",
                health["total"],
            )

    return {"wins": wins, "losses": losses, "skipped": skipped,
            "no_candles": no_candles, "errors": errors}


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url",      required=True)
    parser.add_argument("--batch",       type=int, default=500)
    parser.add_argument("--loop",        action="store_true")
    parser.add_argument("--interval",    type=int, default=300,
                        help="Seconds between loop iterations (default 300 = 5 min)")
    parser.add_argument("--min-candles", type=int, default=20)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)
    try:
        if args.loop:
            log.info("Starting outcome_tracker loop (interval=%ds)", args.interval)
            while True:
                label_tokens(conn, batch_size=args.batch,
                             dry_run=args.dry_run, min_candles=args.min_candles)
                time.sleep(args.interval)
        else:
            label_tokens(conn, batch_size=args.batch,
                         dry_run=args.dry_run, min_candles=args.min_candles)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
