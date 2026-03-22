pub mod instruction_parser;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::config::BotConfig;
use crate::types::TokenSignal;

pub struct YellowstoneScanner {
    config: Arc<BotConfig>,
    signal_tx: broadcast::Sender<TokenSignal>,
}

impl YellowstoneScanner {
    pub fn new(config: Arc<BotConfig>, signal_tx: broadcast::Sender<TokenSignal>) -> Self {
        Self { config, signal_tx }
    }

    pub async fn run(&self) -> Result<()> {
        let url = &self.config.rpc.yellowstone_grpc_url;
        if url.is_empty() || url.starts_with("YOUR_") {
            warn!("Yellowstone gRPC not configured — scanner disabled (Twitter/Telegram signals still active)");
            tokio::time::sleep(std::time::Duration::from_secs(86400)).await;
            return Ok(());
        }

        info!("🔍 Yellowstone scanner: connecting to {}", &url[..url.len().min(40)]);

        // Yellowstone gRPC connection requires the yellowstone-grpc-client crate.
        // When credentials are configured, this will stream all Pump.fun and
        // Raydium launch/swap transactions in real time.
        warn!("Yellowstone scanner: gRPC streaming not yet implemented — use Twitter/Telegram signals");
        let _ = &self.signal_tx; // keep field live; suppress dead_code lint
        tokio::time::sleep(std::time::Duration::from_secs(86400)).await;
        Ok(())
    }
}
