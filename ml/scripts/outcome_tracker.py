"""
outcome_tracker.py — Background job that labels token outcomes every 5 minutes.
Labels ml_label=1 if token hit 2x within 30 min, else 0.
Run: python ml/scripts/outcome_tracker.py
"""
import asyncio, logging, os
from datetime import datetime, timedelta, timezone
import asyncpg, aiohttp

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

BIRDEYE_KEY     = os.environ.get('BIRDEYE_API_KEY', '')
DB_URL          = os.environ.get('DATABASE_URL', 'postgresql://bot:password@localhost:5432/memecoin_bot')
PUMP_THRESHOLD  = 2.0
PUMP_WINDOW_MIN = 30
MIN_LIQ_RATIO   = 0.50

async def main():
    conn = await asyncpg.connect(DB_URL)
    log.info("Outcome tracker started")
    while True:
        try:
            await label_tokens(conn)
            await update_wallet_stats(conn)
        except Exception as e:
            log.error(f"Error: {e}")
        await asyncio.sleep(300)

async def label_tokens(conn):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=35)
    rows = await conn.fetch("""
        SELECT mint, first_seen_at, liquidity_usd FROM tokens
        WHERE ml_label IS NULL AND first_seen_at < $1
          AND first_seen_at > $1 - INTERVAL '48 hours'
        LIMIT 200
    """, cutoff)
    if not rows: return
    log.info(f"Labeling {len(rows)} tokens...")
    async with aiohttp.ClientSession() as session:
        for row in rows:
            try:
                label, outcome, peak = await compute_label(session, row['mint'], row['first_seen_at'])
                await conn.execute("""
                    UPDATE tokens SET ml_label=$1, outcome=$2, peak_multiplier=$3, updated_at=NOW()
                    WHERE mint=$4
                """, label, outcome, peak, row['mint'])
            except Exception as e:
                log.warning(f"Failed to label {row['mint'][:8]}: {e}")

async def compute_label(session, mint, launch_time):
    end = launch_time + timedelta(minutes=PUMP_WINDOW_MIN + 5)
    url = "https://public-api.birdeye.so/defi/ohlcv"
    headers = {"X-API-KEY": BIRDEYE_KEY}
    params = {"address": mint, "type": "1m",
              "time_from": int(launch_time.timestamp()),
              "time_to":   int(end.timestamp())}
    async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        if resp.status != 200: return None, "data_unavailable", 0.0
        data = await resp.json()
    candles = data.get("data", {}).get("items", [])
    if not candles: return 0, "inactive", 0.0
    launch_price = candles[0].get("o", 0)
    if not launch_price: return 0, "invalid", 0.0
    peak_price = max(c.get("h", 0) for c in candles)
    peak_mult  = peak_price / launch_price
    final_price = candles[-1].get("c", 0)
    if final_price < launch_price * 0.3 and peak_mult < PUMP_THRESHOLD:
        return 0, "rug", peak_mult
    if peak_mult >= PUMP_THRESHOLD:
        has_exit = any(c.get("v", 0) > 0 and c.get("c", 0) > launch_price for c in candles)
        if has_exit and peak_mult < 100.0: return 1, "pump", peak_mult
        return 0, "fake_pump", peak_mult
    if peak_mult < 0.5: return 0, "dump", peak_mult
    return 0, "survived_no_pump", peak_mult

async def update_wallet_stats(conn):
    """Update win-rate stats for copy-trade wallets via signals.source_wallet join."""
    await conn.execute("""
        UPDATE copy_wallets cw
        SET
            total_trades   = s.total,
            winning_trades = s.wins,
            win_rate       = CASE WHEN s.total > 0
                                  THEN s.wins::decimal / s.total
                                  ELSE 0
                             END,
            updated_at     = NOW()
        FROM (
            SELECT
                sg.source_wallet,
                COUNT(tr.id)                                        AS total,
                SUM(CASE WHEN tr.pnl_sol > 0 THEN 1 ELSE 0 END)   AS wins
            FROM   signals sg
            JOIN   trades  tr ON tr.mint = sg.mint
            WHERE  sg.source        = 'copy_trade'
              AND  sg.source_wallet IS NOT NULL
              AND  tr.status        = 'closed'
            GROUP  BY sg.source_wallet
        ) s
        WHERE cw.wallet = s.source_wallet
    """)

if __name__ == '__main__':
    asyncio.run(main())
