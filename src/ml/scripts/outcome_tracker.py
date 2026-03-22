"""
outcome_tracker.py — Labels historical token signals as wins or losses
based on post-entry price action, then writes the ml_label column.

BUG FIX (Bug #7): MIN_LIQ_RATIO was defined at the top of the original file
but never referenced anywhere — it was intended to skip labeling tokens whose
liquidity at detection time was so low the price impact would be unreliable.
It is now used in `should_skip_token()` as originally intended.
"""

import argparse
import logging
import time
from datetime import timedelta

import psycopg2
import psycopg2.extras

log = logging.getLogger("outcome_tracker")

# ── Labeling thresholds ───────────────────────────────────────────────────────

WIN_MULTIPLIER    = 1.40   # price must reach 1.40× entry to be a win
LOSS_MULTIPLIER   = 0.75   # price at or below 0.75× entry within observation → loss
OBSERVATION_HOURS = 4      # how long after first_seen_at to observe price action
MIN_CANDLES       = 10     # skip tokens with fewer recorded candles (too sparse)

# FIX (Bug #7): MIN_LIQ_RATIO was a dead constant.  It is now used in
# should_skip_token() to skip tokens where liquidity was so thin that any
# real-world trade would move the price significantly, making the label
# unreliable.  Tokens with liq / mcap < 0.03 are excluded from labeling.
MIN_LIQ_RATIO = 0.03   # minimum liquidity-to-mcap ratio for reliable labeling


# ── Helpers ───────────────────────────────────────────────────────────────────

def should_skip_token(token: dict) -> tuple[bool, str]:
    """Return (True, reason) if this token should not receive a label."""
    liq  = token.get("liquidity_usd_at_detection") or 0.0
    mcap = token.get("market_cap_usd_at_detection") or 0.0

    if liq <= 0 or mcap <= 0:
        return True, "missing_liquidity_or_mcap"

    liq_ratio = liq / mcap
    if liq_ratio < MIN_LIQ_RATIO:
        # FIX: actually use the constant instead of ignoring it
        return True, f"liq_ratio_too_low:{liq_ratio:.4f}<{MIN_LIQ_RATIO}"

    return False, ""


def compute_label(candles: list[dict], entry_price: float) -> int | None:
    """
    Walk candles chronologically after entry.
    Returns 1 (win) if price hits WIN_MULTIPLIER first,
            0 (loss) if it hits LOSS_MULTIPLIER first or time expires,
            None if we cannot determine (too few candles, no price action).
    """
    if not candles or entry_price <= 0:
        return None

    # Sort ascending by timestamp
    candles = sorted(candles, key=lambda c: c["ts"])

    win_threshold  = entry_price * WIN_MULTIPLIER
    loss_threshold = entry_price * LOSS_MULTIPLIER
    hit_win = False
    hit_loss = False

    for c in candles:
        high = c.get("high") or 0.0
        low  = c.get("low")  or 0.0
        if high >= win_threshold:
            hit_win = True
            break
        if low <= loss_threshold:
            hit_loss = True
            break

    if hit_win:
        return 1
    if hit_loss:
        return 0
    # Observation window closed without hitting either threshold — label as loss
    # (price failed to make a meaningful move)
    return 0


# ── Core labeling logic ────────────────────────────────────────────────────────

def label_tokens(conn, batch_size: int = 200, dry_run: bool = False):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Fetch tokens that are old enough to observe and not yet labeled
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
    log.info("Unlabeled tokens to process: %d", len(tokens))

    wins = losses = skipped = errors = 0

    for tok in tokens:
        mint = tok["mint"]

        skip, reason = should_skip_token(tok)
        if skip:
            log.debug("Skipping %s: %s", mint[:8], reason)
            skipped += 1
            continue

        entry_price = tok["price_usd_at_detection"]
        obs_end     = tok["first_seen_at"] + timedelta(hours=OBSERVATION_HOURS)

        try:
            cur.execute("""
                SELECT ts, high, low FROM price_candles_2s
                WHERE mint = %s
                  AND ts BETWEEN %s AND %s
                ORDER BY ts ASC
                LIMIT 2000
            """, (mint, tok["first_seen_at"], obs_end))
            candles = cur.fetchall()
        except Exception as exc:
            log.warning("Candle fetch failed for %s: %s", mint[:8], exc)
            errors += 1
            continue

        if len(candles) < MIN_CANDLES:
            log.debug("Too few candles (%d) for %s — skipping", len(candles), mint[:8])
            skipped += 1
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
                "UPDATE tokens SET ml_label = %s WHERE mint = %s",
                (label, mint),
            )

    if not dry_run:
        conn.commit()

    total = wins + losses
    log.info(
        "Labeling complete | wins=%d losses=%d skipped=%d errors=%d | "
        "win_rate=%.1f%% | dry_run=%s",
        wins, losses, skipped, errors,
        (wins / total * 100 if total else 0),
        dry_run,
    )
    return {"wins": wins, "losses": losses, "skipped": skipped, "errors": errors}


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url",    required=True)
    parser.add_argument("--batch",     type=int, default=200, help="Tokens per run")
    parser.add_argument("--loop",      action="store_true",   help="Run continuously every 60s")
    parser.add_argument("--dry-run",   action="store_true",   help="Compute labels but don't write")
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)
    try:
        if args.loop:
            while True:
                label_tokens(conn, batch_size=args.batch, dry_run=args.dry_run)
                time.sleep(60)
        else:
            label_tokens(conn, batch_size=args.batch, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
