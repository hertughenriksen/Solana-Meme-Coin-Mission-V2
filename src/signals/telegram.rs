use anyhow::{Context, Result};
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::BotConfig;
use crate::types::*;

const SOLANA_ADDR_RE:     &str = r"\b[1-9A-HJ-NP-Za-km-z]{43,44}\b";
const CA_PREFIX_RE:       &str = r"(?:CA|ca|contract|Contract|mint|Mint)\s*[:：]\s*([1-9A-HJ-NP-Za-km-z]{43,44})";
const VELOCITY_WINDOW_SECS: u64 = 60;

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

        info!("📱 Telegram scanner starting...");

        let api_id: i32 = cfg.api_id.parse().context("Telegram API ID must be a number")?;

        // ── grammers fa7692e API ──────────────────────────────────────────────
        // At this commit the public surface is:
        //   Client::connect(InitParams) -> Result<Client>
        //   InitParams { api_id, api_hash, session, .. Default::default() }
        //   session is grammers_session::Session (not a trait — it's a struct)
        //   client.is_authorized() -> Result<bool>
        //   client.resolve_username(&str) -> Result<Option<Chat>>
        //   chat.id() -> i64
        //   client.next_update() -> Result<Update>
        //   Update is an enum with NewMessage(Message), MessageEdited(Message) variants
        //   client.request_login_code(phone, api_id, api_hash) -> Result<LoginToken>
        //   client.sign_in(&LoginToken, &str) -> Result<User, SignInError>
        //   SignInError::PasswordRequired(PasswordToken)
        //   client.check_password(PasswordToken, &str) -> Result<User>
        //   client.session() -> &Session   (saves internally, not to file directly)
        //   Session has no save_to_file; use std::fs::write + session.save()

        use grammers_client::{Client, Config as GrammersConfig, InitParams};
        use grammers_session::Session;

        let session_path = "./secrets/telegram.session";
        let session = if std::path::Path::new(session_path).exists() {
            let data = std::fs::read(session_path)?;
            Session::load(&data).unwrap_or_else(|_| Session::new())
        } else {
            Session::new()
        };

        let client = Client::connect(GrammersConfig {
            session,
            api_id,
            api_hash: cfg.api_hash.clone(),
            params: InitParams {
                catch_up: false,
                ..Default::default()
            },
        }).await.context("Failed to connect to Telegram MTProto")?;

        if !client.is_authorized().await? {
            info!("Telegram: not authorized — starting phone auth flow");
            self.authorize(&client, &cfg.phone, api_id, &cfg.api_hash, session_path).await?;
        } else {
            // Save fresh session data
            self.save_session(&client, session_path)?;
        }

        info!("✅ Telegram authenticated");

        let mut group_states: HashMap<i64, GroupState> = HashMap::new();
        for group_username in &cfg.groups {
            match client.resolve_username(group_username).await {
                Ok(Some(entity)) => {
                    let chat_id = entity.id();
                    group_states.insert(chat_id, GroupState::new());
                    info!("  ✓ Watching group: @{} (id: {})", group_username, chat_id);
                }
                Ok(None) => warn!("  ✗ Group not found: @{}", group_username),
                Err(e)   => warn!("  ✗ Failed to resolve @{}: {}", group_username, e),
            }
        }

        info!("📡 Listening for messages across {} groups", group_states.len());

        loop {
            match client.next_update().await {
                Ok(update) => {
                    use grammers_client::Update;
                    let msg_opt = match update {
                        Update::NewMessage(ref msg) | Update::MessageEdited(ref msg) => {
                            if msg.outgoing() { None }
                            else { Some((msg.chat().id(), msg.text().to_string(), msg.id())) }
                        }
                        _ => None,
                    };

                    if let Some((chat_id, text, _msg_id)) = msg_opt {
                        if text.is_empty() { continue; }
                        if let Some(state) = group_states.get_mut(&chat_id) {
                            let velocity = state.velocity();
                            let min_v    = cfg.min_message_velocity as f64;
                            if velocity >= min_v || cfg.min_message_velocity == 0 {
                                if let Err(e) = self.process_message(&text, chat_id, state, velocity).await {
                                    debug!("Message processing error: {}", e);
                                }
                            }
                        }
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    async fn process_message(
        &self,
        text:           &str,
        _chat_id:       i64,
        state:          &mut GroupState,
        group_velocity: f64,
    ) -> Result<()> {
        if text.len() < 10 { return Ok(()); }

        let mints     = self.extract_mint_addresses(text);
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

    fn save_session(&self, client: &grammers_client::Client, path: &str) -> Result<()> {
        // grammers fa7692e: client.session() returns &Session; Session::save() serialises it
        let session_data = client.session().save();
        std::fs::write(path, &session_data)?;
        Ok(())
    }

    async fn authorize(
        &self,
        client:       &grammers_client::Client,
        phone:        &str,
        api_id:       i32,
        api_hash:     &str,
        session_path: &str,
    ) -> Result<()> {
        use std::io::{self, BufRead, Write};
        // FIX: request_login_code takes (phone, api_id, api_hash) in fa7692e
        let token = client.request_login_code(phone, api_id, api_hash).await?;
        print!("Enter Telegram verification code: ");
        io::stdout().flush()?;
        let mut code = String::new();
        io::stdin().lock().read_line(&mut code)?;

        match client.sign_in(&token, code.trim()).await {
            Ok(_) => {
                info!("Telegram auth successful");
                self.save_session(client, session_path)?;
            }
            Err(grammers_client::SignInError::PasswordRequired(pw_token)) => {
                print!("Enter 2FA password: ");
                io::stdout().flush()?;
                let mut password = String::new();
                io::stdin().lock().read_line(&mut password)?;
                client.check_password(pw_token, password.trim()).await?;
                self.save_session(client, session_path)?;
            }
            Err(e) => anyhow::bail!("Telegram auth failed: {}", e),
        }
        Ok(())
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
