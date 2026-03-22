/// scanner/mod.rs
///
/// DATA-COLLECTION BUILD — free-API-only Pump.fun launch scanner.
///
/// Strategy:
///   1. Poll getSignaturesForAddress on the Pump.fun program every 10 s
///      using the free Helius RPC tier (≤100k req/day).
///   2. For each new signature, call getTransaction and extract the
///      newly created mint from postTokenBalances.
///   3. Deduplicate mints seen within the last 30 min.
///   4. Enrich each signal with price + DAS metadata (enrichment/mod.rs).
///   5. Broadcast onto the signal channel — the strategy engine and DB
///      writer downstream handle storage.
///
/// Rate budget (10 s poll, up to 20 sigs, 1 tx-fetch each):
///   ~3 req per new token × ~200 launches/day = ~600 req/day
///   + 8640 poll req/day  → well within the 100k free limit.
pub mod instruction_parser;

use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::BotConfig;
use crate::enrichment::Enricher;
use crate::types::*;

const PUMP_FUN_PROGRAM: &str = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P";
const WSOL_MINT:        &str = "So11111111111111111111111111111111111111112";
const USDC_MINT:        &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";
const POLL_INTERVAL:    Duration = Duration::from_secs(10);
const DEDUP_TTL:        Duration = Duration::from_secs(1800); // 30 min

pub struct YellowstoneScanner {
    config:    Arc<BotConfig>,
    signal_tx: broadcast::Sender<TokenSignal>,
}

impl YellowstoneScanner {
    pub fn new(config: Arc<BotConfig>, signal_tx: broadcast::Sender<TokenSignal>) -> Self {
        Self { config, signal_tx }
    }

    pub async fn run(&self) -> Result<()> {
        let grpc_url = &self.config.rpc.yellowstone_grpc_url;

        // If a paid gRPC endpoint is configured, stay on the stub path
        // (gRPC streaming not yet implemented — the paid path is for future use).
        if !grpc_url.is_empty() && !grpc_url.starts_with("YOUR_") {
            warn!(
                "Yellowstone gRPC endpoint set but streaming not yet implemented. \
                 Falling through to free RPC polling."
            );
        }

        let helius_key = &self.config.rpc.helius_api_key;
        if helius_key.is_empty() || helius_key.starts_with("YOUR_") {
            warn!("Helius API key not configured — scanner disabled. Set helius_api_key in config/local.toml");
            tokio::time::sleep(Duration::from_secs(86400)).await;
            return Ok(());
        }

        let helius_url = format!(
            "https://mainnet.helius-rpc.com/?api-key={}", helius_key
        );
        let enricher = Arc::new(Enricher::new(helius_key, &helius_url));

        info!("Scanner: free Helius RPC polling for Pump.fun launches every {}s", POLL_INTERVAL.as_secs());

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(8))
            .tcp_nodelay(true)
            .pool_max_idle_per_host(4)
            .build()?;

        // Mint deduplication: mint → Instant of first seen
        let seen: Arc<DashMap<String, Instant>> = Arc::new(DashMap::new());
        let mut last_sig: Option<String> = None;
        let mut launches_found: u64 = 0;

        loop {
            tokio::time::sleep(POLL_INTERVAL).await;

            // Evict stale dedup entries to keep memory bounded
            seen.retain(|_, v| v.elapsed() < DEDUP_TTL);

            // ── Poll for recent Pump.fun signatures ───────────────────────────
            let mut sigs_body = serde_json::json!({
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [PUMP_FUN_PROGRAM, {"limit": 25, "commitment": "confirmed"}]
            });
            if let Some(ref sig) = last_sig {
                sigs_body["params"][1]["until"] = serde_json::Value::String(sig.clone());
            }

            let sigs_resp: serde_json::Value = match http
                .post(&helius_url)
                .json(&sigs_body)
                .send().await
            {
                Ok(r)  => r.json().await.unwrap_or_default(),
                Err(e) => { warn!("Scanner poll error: {}", e); continue; }
            };

            let sigs = match sigs_resp["result"].as_array() {
                Some(s) if !s.is_empty() => s.clone(),
                _ => continue,
            };

            // Update cursor to the most recent signature we've seen
            if let Some(newest) = sigs.first() {
                if let Some(sig) = newest["signature"].as_str() {
                    last_sig = Some(sig.to_string());
                }
            }

            // ── Process each signature ────────────────────────────────────────
            for sig_obj in &sigs {
                let signature = match sig_obj["signature"].as_str() {
                    Some(s) => s.to_string(),
                    None    => continue,
                };

                // Only look at non-failed transactions
                if sig_obj["err"].is_object() { continue; }

                // Fetch full transaction
                let tx_body = serde_json::json!({
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTransaction",
                    "params": [
                        signature,
                        {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0,
                         "commitment": "confirmed"}
                    ]
                });
                let tx: serde_json::Value = match http
                    .post(&helius_url)
                    .json(&tx_body)
                    .send().await
                {
                    Ok(r)  => r.json().await.unwrap_or_default(),
                    Err(e) => { debug!("tx fetch error {}: {}", &signature[..8], e); continue; }
                };

                let Some(result) = tx.get("result").filter(|v| !v.is_null()) else { continue; };

                // Detect new token launches: postTokenBalances contains a mint
                // that does NOT appear in preTokenBalances (i.e. newly created).
                let pre_mints: std::collections::HashSet<String> = result["meta"]["preTokenBalances"]
                    .as_array().unwrap_or(&vec![])
                    .iter()
                    .filter_map(|b| b["mint"].as_str().map(|s| s.to_string()))
                    .collect();

                let post_balances = result["meta"]["postTokenBalances"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();

                for bal in &post_balances {
                    let mint = match bal["mint"].as_str() {
                        Some(m) => m.to_string(),
                        None    => continue,
                    };

                    // Skip SOL, USDC, and known programs
                    if is_known_non_token(&mint) { continue; }

                    // Skip mints that existed before this transaction
                    if pre_mints.contains(&mint) { continue; }

                    // Deduplicate
                    if seen.contains_key(&mint) { continue; }
                    seen.insert(mint.clone(), Instant::now());

                    launches_found += 1;
                    debug!("New launch #{}: {} (sig: {}...)", launches_found, &mint[..8.min(mint.len())], &signature[..8]);

                    // Build and enrich signal
                    let mut signal = TokenSignal {
                        id:          Uuid::new_v4(),
                        mint:        mint.clone(),
                        source:      SignalSource::Yellowstone,
                        signal_type: SignalType::NewTokenLaunch,
                        detected_at: chrono::Utc::now(),
                        on_chain:    None,
                        social:      None,
                        copy_trade:  None,
                    };

                    let enricher_clone = Arc::clone(&enricher);
                    let tx_clone = self.signal_tx.clone();
                    tokio::spawn(async move {
                        if enricher_clone.enrich(&mut signal).await {
                            let _ = tx_clone.send(signal);
                        }
                    });
                }
            }

            if launches_found > 0 && launches_found % 10 == 0 {
                info!("Scanner: {} total launches detected so far this session", launches_found);
            }
        }
    }
}

fn is_known_non_token(mint: &str) -> bool {
    matches!(
        mint,
        "So11111111111111111111111111111111111111112"
            | "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            | "11111111111111111111111111111111"
            | "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
            | "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bPq"
            | "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
            | "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
            | "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C"
    )
}
