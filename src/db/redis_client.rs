use anyhow::Result;
use redis::{aio::ConnectionManager, AsyncCommands, Client};
use crate::types::{FilterResult, Trade};

pub struct RedisClient { conn: ConnectionManager }

impl RedisClient {
    pub async fn connect(url: &str) -> Result<Self> {
        let client = Client::open(url)?;
        let conn = ConnectionManager::new(client).await?;
        Ok(Self { conn })
    }

    pub async fn set_open_position(&self, mint: &str, trade: &Trade) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c.set(format!("position:{}", mint), serde_json::to_string(trade)?).await?;
        // Use INCRBYFLOAT via raw command since AsyncCommands doesn't expose it directly
        let _: () = redis::cmd("INCRBYFLOAT")
            .arg("capital_at_risk")
            .arg(trade.entry_amount_sol)
            .query_async(&mut c)
            .await
            .unwrap_or(());
        Ok(())
    }

    pub async fn remove_open_position(&self, mint: &str) -> Result<()> {
        let mut c = self.conn.clone();
        let key = format!("position:{}", mint);
        if let Ok(json) = c.get::<_, String>(&key).await {
            if let Ok(trade) = serde_json::from_str::<Trade>(&json) {
                let _: () = redis::cmd("INCRBYFLOAT")
                    .arg("capital_at_risk")
                    .arg(-trade.entry_amount_sol)
                    .query_async(&mut c)
                    .await
                    .unwrap_or(());
            }
        }
        let _: () = c.del(&key).await?;
        for suffix in &["tp1_taken", "tp2_taken", "trail_stop"] {
            let _: () = c.del(format!("position:{}:{}", mint, suffix)).await.unwrap_or(());
        }
        Ok(())
    }

    pub async fn has_active_position(&self, mint: &str) -> Result<bool> {
        let mut c = self.conn.clone();
        Ok(c.exists(format!("position:{}", mint)).await?)
    }

    pub async fn get_open_position_count(&self) -> Result<u32> {
        let mut c = self.conn.clone();
        let keys: Vec<String> = c.keys("position:*").await.unwrap_or_default();
        Ok(keys.iter().filter(|k| k.split(':').count() == 2).count() as u32)
    }

    pub async fn get_all_open_positions(&self) -> Result<Vec<Trade>> {
        let mut c = self.conn.clone();
        let keys: Vec<String> = c.keys("position:*").await.unwrap_or_default();
        let mut trades = Vec::new();
        for key in keys.iter().filter(|k| k.split(':').count() == 2) {
            if let Ok(json) = c.get::<_, String>(key).await {
                if let Ok(t) = serde_json::from_str::<Trade>(&json) { trades.push(t); }
            }
        }
        Ok(trades)
    }

    pub async fn get_capital_at_risk_sol(&self) -> Result<f64> {
        let mut c = self.conn.clone();
        let v: Option<String> = c.get("capital_at_risk").await.unwrap_or(None);
        Ok(v.and_then(|s| s.parse().ok()).unwrap_or(0.0))
    }

    pub async fn has_taken_tp1(&self, mint: &str) -> Result<bool> {
        let mut c = self.conn.clone();
        Ok(c.exists(format!("position:{}:tp1_taken", mint)).await?)
    }
    pub async fn mark_tp1_taken(&self, mint: &str) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c.set(format!("position:{}:tp1_taken", mint), "1").await?;
        Ok(())
    }
    pub async fn has_taken_tp2(&self, mint: &str) -> Result<bool> {
        let mut c = self.conn.clone();
        Ok(c.exists(format!("position:{}:tp2_taken", mint)).await?)
    }
    pub async fn mark_tp2_taken(&self, mint: &str) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c.set(format!("position:{}:tp2_taken", mint), "1").await?;
        Ok(())
    }
    pub async fn set_trailing_stop(&self, mint: &str, price: f64) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c.set(format!("position:{}:trail_stop", mint), price.to_string()).await?;
        Ok(())
    }
    pub async fn get_trailing_stop(&self, mint: &str) -> Result<Option<f64>> {
        let mut c = self.conn.clone();
        let v: Option<String> = c.get(format!("position:{}:trail_stop", mint)).await.unwrap_or(None);
        Ok(v.and_then(|s| s.parse().ok()))
    }

    pub async fn get_filter_result(&self, key: &str) -> Result<Option<FilterResult>> {
        let mut c = self.conn.clone();
        let v: Option<String> = c.get(key).await.unwrap_or(None);
        Ok(v.and_then(|s| serde_json::from_str(&s).ok()))
    }
    pub async fn set_filter_result(&self, key: &str, result: &FilterResult, ttl: u64) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c.set_ex(key, serde_json::to_string(result)?, ttl).await?;
        Ok(())
    }

    pub async fn get_circuit_breaker_active(&self) -> Result<bool> {
        let mut c = self.conn.clone();
        Ok(c.exists("circuit_breaker").await?)
    }
    pub async fn set_circuit_breaker_active(&self, duration_secs: u32) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c.set_ex("circuit_breaker", "1", duration_secs as u64).await?;
        Ok(())
    }

    pub async fn get_network_congestion_multiplier(&self) -> Result<f64> {
        let mut c = self.conn.clone();
        let v: Option<String> = c.get("congestion:multiplier").await.unwrap_or(None);
        Ok(v.and_then(|s| s.parse().ok()).unwrap_or(1.0))
    }
    pub async fn get_cached_price(&self, mint: &str) -> Option<f64> {
        let mut c = self.conn.clone();
        c.get::<_, String>(format!("price:{}", mint)).await.ok()
            .and_then(|s| s.parse().ok())
    }
    pub async fn set_cached_price(&self, mint: &str, price: f64) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c.set_ex(format!("price:{}", mint), price.to_string(), 2).await?;
        Ok(())
    }
    pub async fn increment_signals_scanned(&self) -> Result<u64> {
        let mut c = self.conn.clone();
        Ok(c.incr("signals:scanned", 1u64).await.unwrap_or(0))
    }
    pub async fn increment_signals_filtered_out(&self) -> Result<u64> {
        let mut c = self.conn.clone();
        Ok(c.incr("signals:filtered_out", 1u64).await.unwrap_or(0))
    }
    pub async fn get_signal_counts(&self) -> Result<(u64, u64)> {
        let mut c = self.conn.clone();
        let s: u64 = c.get::<_, String>("signals:scanned").await.ok()
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        let f: u64 = c.get::<_, String>("signals:filtered_out").await.ok()
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        Ok((s, f))
    }
    pub async fn record_bundle_submitted(&self) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c.incr("jito:submitted", 1u64).await?;
        Ok(())
    }
    pub async fn record_bundle_accepted(&self) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c.incr("jito:accepted", 1u64).await?;
        Ok(())
    }
    pub async fn get_recent_tip_acceptance_rate(&self) -> f64 {
        let mut c = self.conn.clone();
        let accepted: u64 = c.get::<_, String>("jito:accepted").await.ok()
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        let total: u64 = c.get::<_, String>("jito:submitted").await.ok()
            .and_then(|v| v.parse().ok()).unwrap_or(1);
        accepted as f64 / total as f64
    }
}
