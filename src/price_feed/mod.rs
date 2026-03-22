/// price_feed/mod.rs
///
/// DATA-COLLECTION BUILD change:
///   In addition to writing prices to Redis (for the position manager),
///   every tick now also writes a price_candles_2s row to PostgreSQL
///   for every tracked token.  This is what outcome_tracker.py reads
///   to label tokens as wins or losses — without these rows there is
///   nothing to train on.
///
///   The DB write is fire-and-forget (tokio::spawn) so it never blocks
///   the 2-second polling loop.
use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

use crate::db::{Database, RedisClient};
use crate::types::Candle;

const JUPITER_PRICE_URL: &str = "https://lite-api.jup.ag/price/v2";
const SOL_MINT:          &str = "So11111111111111111111111111111111111111112";

pub struct PriceFeed {
    redis:  Arc<RedisClient>,
    db:     Arc<Database>,
    http:   reqwest::Client,
    /// Per-mint last price, used to synthesise OHLC from tick data.
    last_prices: std::sync::Arc<dashmap::DashMap<String, f64>>,
}

impl PriceFeed {
    pub fn new(redis: Arc<RedisClient>, db: Arc<Database>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .tcp_nodelay(true)
            .build()
            .expect("PriceFeed HTTP client");
        Self {
            redis,
            db,
            http,
            last_prices: Arc::new(dashmap::DashMap::new()),
        }
    }

    pub async fn run(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            if let Err(e) = self.tick().await {
                warn!("PriceFeed error: {}", e);
            }
        }
    }

    async fn tick(&self) -> Result<()> {
        let positions = self.redis.get_all_open_positions().await?;
        let mut mints: Vec<String> = positions.into_iter().map(|p| p.mint).collect();
        if !mints.contains(&SOL_MINT.to_string()) {
            mints.push(SOL_MINT.to_string());
        }
        mints.dedup();

        if mints.is_empty() { return Ok(()); }

        // Jupiter accepts up to ~100 mints per request
        let ids = mints.join(",");
        let url = format!("{}?ids={}", JUPITER_PRICE_URL, ids);

        let resp: serde_json::Value = self.http.get(&url).send().await?.json().await?;

        let data = match resp.get("data").and_then(|d| d.as_object()) {
            Some(d) => d.clone(),
            None    => {
                warn!("PriceFeed: unexpected response: {}", resp);
                return Ok(());
            }
        };

        let now = chrono::Utc::now();

        for (mint, price_obj) in &data {
            let price: f64 = if let Some(p) = price_obj["price"].as_f64() {
                p
            } else if let Some(s) = price_obj["price"].as_str() {
                match s.parse::<f64>() { Ok(p) => p, Err(_) => continue }
            } else {
                continue
            };

            if price <= 0.0 { continue; }

            let key = if mint == SOL_MINT { "SOL".to_string() } else { mint.clone() };

            // ── Redis write (position manager reads this) ─────────────────────
            if let Err(e) = self.redis.set_cached_price(&key, price).await {
                warn!("PriceFeed Redis write failed for {}: {}", &key[..key.len().min(8)], e);
            }

            // ── Build a synthetic 2s OHLC candle from successive ticks ────────
            // We don't have real bid/ask spread on the free API, so we treat
            // each price as the close and derive OHLC from the last two ticks.
            // This is sufficient for the outcome_tracker to label wins/losses.
            let last = self.last_prices.get(&key).map(|v| *v).unwrap_or(price);
            self.last_prices.insert(key.clone(), price);

            let (open, high, low, close) = (last, price.max(last), price.min(last), price);

            let candle = Candle {
                timestamp:  now,
                open,
                high,
                low,
                close,
                volume:     0.0, // volume not available from Jupiter price API
                buy_count:  0,
                sell_count: 0,
            };

            // ── PostgreSQL write (outcome_tracker reads this) ─────────────────
            // Fire-and-forget: never block the price loop on a DB write.
            let db    = Arc::clone(&self.db);
            let mint2 = key.clone();
            tokio::spawn(async move {
                if let Err(e) = db.write_price_candle(&mint2, &candle).await {
                    debug!("PriceFeed candle write failed for {}: {}", &mint2[..mint2.len().min(8)], e);
                }
            });

            debug!("price {:>8} = ${:.8}", &key[..key.len().min(8)], price);
        }

        Ok(())
    }
}
