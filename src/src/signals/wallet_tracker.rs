use anyhow::Result;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::config::BotConfig;
use crate::types::TokenSignal;

pub struct WalletTracker {
    config: Arc<BotConfig>,
    signal_tx: broadcast::Sender<TokenSignal>,
}

impl WalletTracker {
    pub fn new(config: Arc<BotConfig>, signal_tx: broadcast::Sender<TokenSignal>) -> Self {
        Self { config, signal_tx }
    }

    pub async fn run(&self) -> Result<()> {
        let wallets = &self.config.strategy.copy_wallets;
        if wallets.is_empty() {
            info!("No copy-trade wallets configured — wallet tracker idle");
            tokio::time::sleep(std::time::Duration::from_secs(86400)).await;
            return Ok(());
        }

        info!("👀 Wallet tracker: monitoring {} wallets", wallets.len());
        for w in wallets {
            info!("   → {}…{}", &w[..4.min(w.len())], &w[w.len().saturating_sub(4)..]);
        }

        let url = &self.config.rpc.yellowstone_grpc_url;
        if url.is_empty() || url.starts_with("YOUR_") {
            warn!("Yellowstone gRPC not configured — wallet tracker disabled");
            tokio::time::sleep(std::time::Duration::from_secs(86400)).await;
            return Ok(());
        }

        warn!("Wallet tracker: gRPC not yet implemented — configure Yellowstone gRPC to enable");
        let _ = &self.signal_tx;
        tokio::time::sleep(std::time::Duration::from_secs(86400)).await;
        Ok(())
    }
}
