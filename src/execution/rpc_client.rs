use anyhow::{Context, Result};
use parking_lot::RwLock;
use reqwest::Client;
use serde_json::{json, Value};
use solana_sdk::hash::Hash;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

use crate::config::BotConfig;

struct BlockhashCache {
    blockhash: Hash,
    refreshed_at: Instant,
}

impl BlockhashCache {
    fn is_stale(&self) -> bool {
        self.refreshed_at.elapsed() > Duration::from_secs(15)
    }
}

pub struct SolanaRpcClient {
    http: Client,
    config: Arc<BotConfig>,
    cache: Arc<RwLock<Option<BlockhashCache>>>,
}

impl SolanaRpcClient {
    pub fn new(config: Arc<BotConfig>) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(8))
            .tcp_nodelay(true)
            .build()
            .expect("HTTP client build failed");
        Self { http, config, cache: Arc::new(RwLock::new(None)) }
    }

    pub fn spawn_blockhash_refresher(self: &Arc<Self>) {
        let client = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(15));
            loop {
                interval.tick().await;
                if let Ok((hash, _)) = client.fetch_blockhash_from_rpc().await {
                    *client.cache.write() = Some(BlockhashCache {
                        blockhash: hash,
                        refreshed_at: Instant::now(),
                    });
                    debug!("Blockhash refreshed: {}", hash);
                }
            }
        });
    }

    pub async fn get_latest_blockhash(&self) -> Result<Hash> {
        {
            let guard = self.cache.read();
            if let Some(ref c) = *guard {
                if !c.is_stale() { return Ok(c.blockhash); }
            }
        }
        let (hash, _) = self.fetch_blockhash_from_rpc().await?;
        *self.cache.write() = Some(BlockhashCache {
            blockhash: hash,
            refreshed_at: Instant::now(),
        });
        Ok(hash)
    }

    async fn fetch_blockhash_from_rpc(&self) -> Result<(Hash, u64)> {
        let body = json!({
            "jsonrpc": "2.0", "id": 1,
            "method": "getLatestBlockhash",
            "params": [{"commitment": "processed"}],
        });
        let resp     = self.call_with_failover(body).await?;
        let hash_str = resp["result"]["value"]["blockhash"]
            .as_str().context("No blockhash in response")?;
        let slot     = resp["result"]["value"]["lastValidBlockHeight"].as_u64().unwrap_or(0);
        let hash     = Hash::from_str(hash_str).context("Invalid blockhash")?;
        Ok((hash, slot))
    }

    pub async fn get_token_balance(&self, owner: &str, mint: &str) -> Result<u64> {
        let body = json!({
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [owner, {"mint": mint}, {"encoding": "jsonParsed", "commitment": "processed"}],
        });
        let resp     = self.call_with_failover(body).await?;
        let accounts = resp["result"]["value"].as_array().context("No accounts in response")?;
        if accounts.is_empty() { return Ok(0); }
        let amount = accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"]
            .as_str().unwrap_or("0");
        Ok(amount.parse::<u64>().unwrap_or(0))
    }

    pub async fn get_sol_balance(&self, address: &str) -> Result<u64> {
        let body = json!({
            "jsonrpc": "2.0", "id": 1,
            "method": "getBalance",
            "params": [address, {"commitment": "processed"}],
        });
        let resp = self.call_with_failover(body).await?;
        Ok(resp["result"]["value"].as_u64().unwrap_or(0))
    }

    pub async fn confirm_transaction(&self, signature: &str, timeout_secs: u64) -> Result<bool> {
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        while Instant::now() < deadline {
            let body = json!({
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignatureStatuses",
                "params": [[signature], {"searchTransactionHistory": false}],
            });
            if let Ok(resp) = self.call_with_failover(body).await {
                if let Some(status) = resp["result"]["value"][0].as_object() {
                    let conf = status.get("confirmationStatus")
                        .and_then(|v| v.as_str()).unwrap_or("");
                    if conf == "confirmed" || conf == "finalized" {
                        if status.get("err").map(|e| !e.is_null()).unwrap_or(false) {
                            return Err(anyhow::anyhow!("Transaction failed on-chain"));
                        }
                        return Ok(true);
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        }
        Ok(false)
    }

    pub async fn get_multiple_accounts(&self, pubkeys: &[&str]) -> Result<Vec<Option<Value>>> {
        let body = json!({
            "jsonrpc": "2.0", "id": 1,
            "method": "getMultipleAccounts",
            "params": [pubkeys, {"encoding": "base64", "commitment": "processed"}],
        });
        let resp = self.call_with_failover(body).await?;
        let list = resp["result"]["value"]
            .as_array().context("No value array")?
            .iter()
            .map(|v| if v.is_null() { None } else { Some(v.clone()) })
            .collect();
        Ok(list)
    }

    pub async fn get_asset_info(&self, mint: &str) -> Result<Value> {
        let url = format!(
            "https://mainnet.helius-rpc.com/?api-key={}",
            self.config.rpc.helius_api_key,
        );
        self.rpc_call(&url, json!({
            "jsonrpc": "2.0", "id": 1,
            "method": "getAsset",
            "params": {"id": mint},
        })).await
    }

    pub async fn get_token_largest_accounts(&self, mint: &str) -> Result<Vec<(String, u64)>> {
        let body = json!({
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenLargestAccounts",
            "params": [mint, {"commitment": "finalized"}],
        });
        let resp = self.call_with_failover(body).await?;
        let holders = resp["result"]["value"]
            .as_array().unwrap_or(&vec![])
            .iter()
            .filter_map(|h| {
                let address = h["address"].as_str()?.to_string();
                let amount  = h["amount"].as_str()?.parse::<u64>().ok()?;
                Some((address, amount))
            })
            .collect();
        Ok(holders)
    }

    async fn call_with_failover(&self, body: Value) -> Result<Value> {
        for endpoint in self.endpoints() {
            match self.rpc_call(&endpoint, body.clone()).await {
                Ok(v)  => return Ok(v),
                Err(e) => warn!("RPC {} error: {}", endpoint, e),
            }
        }
        anyhow::bail!("All RPC endpoints failed")
    }

    async fn rpc_call(&self, url: &str, body: Value) -> Result<Value> {
        let resp = self.http.post(url).json(&body).send().await?.json::<Value>().await?;
        if let Some(err) = resp.get("error") { anyhow::bail!("RPC error: {}", err); }
        Ok(resp)
    }

    fn endpoints(&self) -> Vec<String> {
        let cfg = &self.config.rpc;
        let mut eps = vec![
            format!("https://mainnet.helius-rpc.com/?api-key={}", cfg.helius_api_key),
        ];
        if !cfg.quicknode_url.starts_with("YOUR_") {
            eps.push(cfg.quicknode_url.clone());
        }
        if !cfg.shyft_api_key.starts_with("YOUR_") {
            eps.push(format!("https://rpc.shyft.to/?api_key={}", cfg.shyft_api_key));
        }
        eps
    }
}
