use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::BotConfig;
use crate::types::*;

const SOLANA_ADDR_RE:       &str = r"\b[1-9A-HJ-NP-Za-km-z]{43,44}\b";
const CA_PREFIX_RE:         &str = r"(?:CA|ca|contract|Contract|mint|Mint)\s*[:：]\s*([1-9A-HJ-NP-Za-km-z]{43,44})";
const VELOCITY_WINDOW_SECS: u64  = 60;

pub struct TelegramScanner {
    config:       Arc<BotConfig>,
    signal_tx:    broadcast::Sender<TokenSignal>,
    address_re:   Regex,
    ca_prefix_re: Regex,
}

struct GroupState {
    message_timestamps: std::collections::VecDeque<std::time::Instant>,
    recent_mints:       HashMap<String, std::time::Instant>,
}

impl GroupState {
    fn new() -> Self {
        Self {
            message_timestamps: std::collections::VecDeque::new(),
            recent_mints:       HashMap::new(),
        }
    }

    fn velocity(&mut self) -> f64 {
        let now    = std::time::Instant::now();
        let cutoff = now - std::time::Duration::from_secs(VELOCITY_WINDOW_SECS);
        while self.message_timestamps.front().map(|&t| t < cutoff).unwrap_or(false) {
            self.message_timestamps.pop_front();
        }
        self.message_timestamps.push_back(now);
        self.message_timestamps.len() as f64
    }

    fn has_seen_mint_recently(&mut self, mint: &str) -> bool {
        let now    = std::time::Instant::now();
        let cutoff = now - std::time::Duration::from_secs(300);
        self.recent_mints.retain(|_, &mut t| t > cutoff);
        if self.recent_mints.contains_key(mint) {
            true
        } else {
            self.recent_mints.insert(mint.to_string(), now);
            false
        }
    }
}

impl TelegramScanner {
    pub fn new(config: Arc<BotConfig>, signal_tx: broadcast::Sender<TokenSignal>) -> Self {
        Self {
            config,
            signal_tx,
            address_re:   Regex::new(SOLANA_ADDR_RE).unwrap(),
            ca_prefix_re: Regex::new(CA_PREFIX_RE).unwrap(),
        }
    }

    pub async fn run(&self) -> Result<()> {
        let cfg = &self.config.signals.telegram;

        if cfg.api_id.is_empty() || cfg.api_id == "YOUR_TELEGRAM_API_ID" {
            warn!("Telegram API credentials not configured — scanner disabled");
            tokio::time::sleep(std::time::Duration::from_secs(86400)).await;
            return Ok(());
        }

        // ── grammers library API compatibility note ────────────────────────────
        //
        // The grammers repository (github.com/Lonami/grammers) is currently
        // mid-rewrite at commit fa7692e (its HEAD).  At that commit the public
        // connection API (Client::connect, Config, InitParams, Session struct,
        // client.session().save(), Update enum at crate root) does not yet exist
        // — those types are internal only.  Cargo resolves to this commit whether
        // or not a `rev` pin is specified because it IS the branch HEAD.
        //
        // Until the grammers authors stabilise the 0.8 public API, Telegram
        // MTProto scanning is disabled at compile time to keep the build green.
        // Twitter and Yellowstone gRPC signals continue to work normally.
        //
        // To re-enable Telegram once grammers ships a stable release:
        //   1. Add `tag = "v0.8.x"` (or the released tag) to the grammers deps
        //      in Cargo.toml.
        //   2. Restore the grammers-based connection code below.
        //   3. Run `cargo update` to fetch the new revision.

        warn!(
            "Telegram scanner: grammers library is at a pre-release commit with no \
             public connection API — scanner disabled. \
             Twitter and Yellowstone signals remain active."
        );
        info!(
            "Telegram re-enable: once grammers ships a stable 0.8 tag, update \
             Cargo.toml with tag = \"v0.8.x\" and restore the connection code."
        );

        // Sleep for 24 h before the caller retries, to avoid a tight log-spam loop.
        tokio::time::sleep(std::time::Duration::from_secs(86400)).await;
        Ok(())
    }

    // ── Internal helpers kept intact for when grammers API stabilises ─────────

    async fn process_message(
        &self,
        text:           &str,
        _chat_id:       i64,
        state:          &mut GroupState,
        group_velocity: f64,
    ) -> Result<()> {
        if text.len() < 10 { return Ok(()); }

        let mints = self.extract_mint_addresses(text);
        if mints.is_empty() { return Ok(()); }

        let sentiment = self.score_sentiment(text);

        for mint in mints {
            if is_known_program(&mint) { continue; }
            if state.has_seen_mint_recently(&mint) { continue; }

            let signal = TokenSignal {
                id: Uuid::new_v4(),
                mint: mint.clone(),
                source: SignalSource::Telegram,
                signal_type: if group_velocity > 20.0 {
                    SignalType::CoordinatedMention
                } else {
                    SignalType::SentimentSpike
                },
                detected_at: chrono::Utc::now(),
                on_chain: None,
                social: Some(SocialData {
                    twitter_mentions_5m: 0, twitter_mentions_1h: 0,
                    telegram_mentions_5m: 1, telegram_mentions_1h: 1,
                    sentiment_score: sentiment,
                    sentiment_acceleration: (group_velocity / 10.0).min(1.0),
                    kol_mention: false,
                    kol_names: vec![],
                    message_samples: vec![text.to_string()],
                }),
                copy_trade: None,
            };

            info!(
                "📱 Telegram signal: {} | velocity: {:.0}/min | sentiment: {:.2}",
                &mint[..8.min(mint.len())], group_velocity, sentiment,
            );
            let _ = self.signal_tx.send(signal);
        }
        Ok(())
    }

    fn extract_mint_addresses(&self, text: &str) -> Vec<String> {
        let mut mints = Vec::new();
        for cap in self.ca_prefix_re.captures_iter(text) {
            if let Some(m) = cap.get(1) { mints.push(m.as_str().to_string()); }
        }
        if mints.is_empty() {
            for m in self.address_re.find_iter(text) {
                let addr = m.as_str().to_string();
                if !mints.contains(&addr) { mints.push(addr); }
            }
        }
        mints
    }

    fn score_sentiment(&self, text: &str) -> f64 {
        let t = text.to_lowercase();
        let bull: f64 = [
            "gem","launch","early","buy","pump","moon","alpha","call","fire","based",
            "locked","100x","safe","legit","🚀","💎","🔥","✅","degen","ape","strong",
        ].iter().filter(|&&s| t.contains(s)).count() as f64;
        let bear: f64 = [
            "rug","scam","dump","avoid","honeypot","bot","bundled",
            "sniper","🚨","⚠️","danger","fake","exit","sell",
        ].iter().filter(|&&s| t.contains(s)).count() as f64;
        (0.5 + (bull - bear * 1.5) * 0.1).clamp(0.0, 1.0)
    }
}

fn is_known_program(addr: &str) -> bool {
    matches!(
        addr,
        "11111111111111111111111111111111"
            | "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
            | "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
            | "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
            | "So11111111111111111111111111111111111111112"
            | "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    )
}
