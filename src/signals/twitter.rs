use anyhow::{Context, Result};
use futures::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::BotConfig;
use crate::types::*;

const SOLANA_ADDRESS_REGEX: &str = r"\b[1-9A-HJ-NP-Za-km-z]{43,44}\b";
const CA_PREFIX_REGEX: &str = r"(?:CA|ca|contract|Contract|mint|Mint)\s*[:：]\s*([1-9A-HJ-NP-Za-km-z]{43,44})";

pub struct TwitterScanner {
    config: Arc<BotConfig>,
    signal_tx: broadcast::Sender<TokenSignal>,
    http: reqwest::Client,
    address_re: Regex,
    ca_prefix_re: Regex,
}

#[derive(Debug, Deserialize)]
struct StreamEvent {
    data: Option<TweetData>,
    matching_rules: Option<Vec<MatchingRule>>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct TweetData {
    id: String,
    text: String,
    author_id: Option<String>,
    #[serde(rename = "public_metrics")]
    metrics: Option<TweetMetrics>,
    #[serde(rename = "referenced_tweets")]
    referenced: Option<Vec<ReferencedTweet>>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct TweetMetrics { like_count: u64, retweet_count: u64, reply_count: u64 }

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct MatchingRule { id: String, tag: Option<String> }

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ReferencedTweet { #[serde(rename = "type")] tweet_type: String, id: String }

#[derive(Debug, Serialize)]
struct StreamRule { value: String, tag: String }

#[derive(Debug, Serialize)]
struct AddRulesRequest { add: Vec<StreamRule> }

#[derive(Debug, Serialize)]
struct DeleteRulesRequest { delete: DeleteRulesIds }

#[derive(Debug, Serialize)]
struct DeleteRulesIds { ids: Vec<String> }

impl TwitterScanner {
    pub fn new(config: Arc<BotConfig>, signal_tx: broadcast::Sender<TokenSignal>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(90))
            .tcp_keepalive(std::time::Duration::from_secs(30))
            .build().expect("HTTP client build failed");
        Self {
            config, signal_tx, http,
            address_re: Regex::new(SOLANA_ADDRESS_REGEX).unwrap(),
            ca_prefix_re: Regex::new(CA_PREFIX_REGEX).unwrap(),
        }
    }

    pub async fn run(&self) -> Result<()> {
        let bearer = &self.config.signals.twitter.bearer_token;
        if bearer.is_empty() || bearer == "YOUR_TWITTER_BEARER_TOKEN" {
            warn!("Twitter bearer token not configured — scanner disabled");
            tokio::time::sleep(std::time::Duration::from_secs(86400)).await;
            return Ok(());
        }

        info!("🐦 Twitter scanner starting...");
        self.sync_stream_rules().await?;

        let stream_url = "https://api.twitter.com/2/tweets/search/stream";
        let params = [
            ("tweet.fields", "author_id,public_metrics,referenced_tweets,created_at"),
            ("expansions", "author_id"),
            ("user.fields", "public_metrics,verified"),
        ];

        let resp = self.http.get(stream_url).bearer_auth(bearer).query(&params)
            .send().await.context("Failed to connect to Twitter stream")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Twitter stream error {}: {}", status, body);
        }

        info!("✅ Connected to Twitter filtered stream");
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Stream read error")?;
            let text = String::from_utf8_lossy(&chunk);
            let text = text.trim();
            if text.is_empty() { continue; }

            match serde_json::from_str::<StreamEvent>(text) {
                Ok(event) => {
                    if let Some(tweet) = event.data {
                        if let Err(e) = self.process_tweet(tweet, &event.matching_rules).await {
                            debug!("Tweet processing error: {}", e);
                        }
                    }
                }
                Err(e) => {
                    debug!("Parse error: {} | raw: {}", e, &text[..text.len().min(200)]);
                }
            }
        }

        warn!("Twitter stream ended — will reconnect");
        Ok(())
    }

    async fn process_tweet(&self, tweet: TweetData, _rules: &Option<Vec<MatchingRule>>) -> Result<()> {
        let cfg = &self.config.signals.twitter;

        if let Some(ref refs) = tweet.referenced {
            if refs.iter().any(|r| r.tweet_type == "retweeted") { return Ok(()); }
        }

        let text      = &tweet.text;
        let author_id = tweet.author_id.as_deref().unwrap_or("");
        let is_kol    = cfg.kol_accounts.iter().any(|id| id == author_id);
        let mints     = self.extract_mint_addresses(text);

        if mints.is_empty() {
            if !is_kol || !self.has_trigger_keywords(text) { return Ok(()); }
            return Ok(());
        }

        let sentiment = self.score_sentiment(text);
        if sentiment < cfg.sentiment_score_threshold && !is_kol { return Ok(()); }

        for mint in mints {
            if is_known_program_address(&mint) { continue; }
            let signal = TokenSignal {
                id: Uuid::new_v4(), mint: mint.clone(),
                source: SignalSource::Twitter,
                signal_type: SignalType::SentimentSpike,
                detected_at: chrono::Utc::now(),
                on_chain: None,
                social: Some(SocialData {
                    twitter_mentions_5m: 1, twitter_mentions_1h: 1,
                    telegram_mentions_5m: 0, telegram_mentions_1h: 0,
                    sentiment_score: sentiment, sentiment_acceleration: 0.0,
                    kol_mention: is_kol,
                    kol_names: if is_kol { vec![author_id.to_string()] } else { vec![] },
                    message_samples: vec![text.clone()],
                }),
                copy_trade: None,
            };
            info!("🐦 Twitter signal: {} | KOL: {} | sentiment: {:.2}", &mint[..8.min(mint.len())], is_kol, sentiment);
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
        let bull: f64 = ["just launched","new gem","100x","moonshot","lp locked","mint disabled",
            "based dev","kyc","early","gem","lfg","pump","huge","massive","fire","🚀","💎","🔥",
            "bullish","alpha","call","aped","buying"].iter().filter(|&&s| t.contains(s)).count() as f64;
        let bear: f64 = ["rug","scam","honeypot","avoid","dump","sell","warning",
            "🚨","⚠️","fake","fraud","exit","beware","bundled","sniper"].iter().filter(|&&s| t.contains(s)).count() as f64;
        (0.5 + (bull - bear * 1.5) * 0.1).clamp(0.0, 1.0)
    }

    fn has_trigger_keywords(&self, text: &str) -> bool {
        let t = text.to_lowercase();
        self.config.signals.twitter.trigger_keywords.iter().any(|kw| t.contains(&kw.to_lowercase()))
    }

    async fn sync_stream_rules(&self) -> Result<()> {
        let bearer    = &self.config.signals.twitter.bearer_token;
        let rules_url = "https://api.twitter.com/2/tweets/search/stream/rules";

        let existing: serde_json::Value = self.http.get(rules_url).bearer_auth(bearer)
            .send().await?.json().await?;

        if let Some(data) = existing["data"].as_array() {
            if !data.is_empty() {
                let ids: Vec<String> = data.iter()
                    .filter_map(|r| r["id"].as_str().map(|s| s.to_string())).collect();
                let delete_req = DeleteRulesRequest { delete: DeleteRulesIds { ids } };
                self.http.post(rules_url).bearer_auth(bearer).json(&delete_req).send().await?;
            }
        }

        let cfg = &self.config.signals.twitter;
        let mut rules = Vec::new();

        if !cfg.kol_accounts.is_empty() {
            let kol_clause = cfg.kol_accounts.iter()
                .map(|id| format!("from:{}", id)).collect::<Vec<_>>().join(" OR ");
            rules.push(StreamRule {
                value: format!("({}) has:links -is:retweet", kol_clause),
                tag: "kol_accounts".into(),
            });
        }

        let keyword_clause = cfg.trigger_keywords.iter()
            .map(|k| format!("\"{}\"", k)).collect::<Vec<_>>().join(" OR ");
        rules.push(StreamRule {
            value: format!("({}) lang:en -is:retweet", keyword_clause),
            tag: "trigger_keywords".into(),
        });
        rules.push(StreamRule {
            value: "\"CA:\" OR \"contract:\" OR \"just launched\" lang:en -is:retweet".into(),
            tag: "ca_pattern".into(),
        });

        let add_req = AddRulesRequest { add: rules };
        let resp: serde_json::Value = self.http.post(rules_url)
            .bearer_auth(bearer).json(&add_req).send().await?.json().await?;
        let rule_count = resp["data"].as_array().map(|a| a.len()).unwrap_or(0);
        info!("✅ Twitter stream rules set ({} rules active)", rule_count);
        Ok(())
    }
}

fn is_known_program_address(addr: &str) -> bool {
    matches!(addr,
        "11111111111111111111111111111111" |
        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" |
        "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P" |
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8" |
        "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C" |
        "So11111111111111111111111111111111111111112" |
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    )
}
