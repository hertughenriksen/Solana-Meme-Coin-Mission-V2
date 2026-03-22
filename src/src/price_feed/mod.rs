/// price_feed/mod.rs — Polls Jupiter price API for all open positions + SOL/USD.
/// Writes results to Redis (key: `price:<mint>`) every 2 seconds so that the
/// position manager and execution engine always have fresh price data.
use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

use crate::db::RedisClient;

const JUPITER_PRICE_URL: &str = "https://lite-api.jup.ag/price/v2";
const SOL_MINT: &str = "So11111111111111111111111111111111111111112";

pub struct PriceFeed {
    redis: Arc<RedisClient>,
    http: reqwest::Client,
}

impl PriceFeed {
    pub fn new(redis: Arc<RedisClient>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .tcp_nodelay(true)
            .build()
            .expect("PriceFeed: HTTP client build failed");
        Self { redis, http }
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

        let ids = mints.join(",");
        let url = format!("{}?ids={}", JUPITER_PRICE_URL, ids);

        let resp: serde_json::Value = self.http.get(&url).send().await?.json().await?;

        let data = match resp.get("data").and_then(|d| d.as_object()) {
            Some(d) => d.clone(),
            None => {
                warn!("PriceFeed: unexpected response shape: {}", resp);
                return Ok(());
            }
        };

        for (mint, price_obj) in &data {
            let price: f64 = if let Some(p) = price_obj["price"].as_f64() {
                p
            } else if let Some(s) = price_obj["price"].as_str() {
                match s.parse::<f64>() {
                    Ok(p) => p,
                    Err(_) => continue,
                }
            } else {
                continue;
            };

            if price <= 0.0 { continue; }

            let key = if mint == SOL_MINT { "SOL".to_string() } else { mint.clone() };
            if let Err(e) = self.redis.set_cached_price(&key, price).await {
                warn!("PriceFeed: failed to write price for {}: {}", &key[..key.len().min(8)], e);
            }
            debug!("price {:>8} = ${:.8}", &key[..key.len().min(8)], price);
        }

        Ok(())
    }
}
