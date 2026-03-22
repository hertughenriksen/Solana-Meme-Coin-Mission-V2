use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSignal {
    pub id: Uuid,
    pub mint: String,
    pub source: SignalSource,
    pub signal_type: SignalType,
    pub detected_at: DateTime<Utc>,
    pub on_chain: Option<OnChainData>,
    pub social: Option<SocialData>,
    pub copy_trade: Option<CopyTradeData>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalSource { Yellowstone, Twitter, Telegram, CopyTrade, Combined }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalType { NewTokenLaunch, LiquidityAdded, SmartWalletBuy, SentimentSpike, CoordinatedMention }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainData {
    pub pool_address: String,
    pub dex: DexType,
    pub liquidity_usd: f64,
    pub market_cap_usd: f64,
    pub token_age_seconds: u64,
    pub deployer_wallet: String,
    pub deployer_wallet_age_days: u64,
    pub deployer_previous_tokens: Vec<PreviousToken>,
    pub mint_authority_disabled: bool,
    pub freeze_authority_disabled: bool,
    pub lp_locked: bool,
    pub lp_lock_days: Option<u32>,
    pub lp_lock_pct: Option<f64>,
    pub dev_holding_pct: f64,
    pub top_10_holder_pct: f64,
    pub sniper_concentration_pct: f64,
    pub buy_count_1h: u32,
    pub sell_count_1h: u32,
    pub buy_count_24h: u32,
    pub sell_count_24h: u32,
    pub price_usd: f64,
    pub price_change_5m_pct: f64,
    pub price_change_1h_pct: f64,
    pub volume_usd_1h: f64,
    pub price_candles_2s: Vec<Candle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub buy_count: u32,
    pub sell_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviousToken {
    pub mint: String,
    pub launched_at: DateTime<Utc>,
    pub outcome: TokenOutcome,
    pub peak_multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TokenOutcome {
    Rug,
    Honeypot,
    Inactive,
    Survived,
    Graduated,
    Pump,
    FakePump,
    Dump,
    SurvivedNoPump,
    DataUnavailable,
    Invalid,
}

impl TokenOutcome {
    pub fn is_negative(&self) -> bool {
        matches!(
            self,
            TokenOutcome::Rug
                | TokenOutcome::Honeypot
                | TokenOutcome::FakePump
                | TokenOutcome::Dump
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DexType {
    PumpFun, PumpSwap, RaydiumAMM, RaydiumCPMM, RaydiumCLMM, Meteora, Orca, Jupiter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialData {
    pub twitter_mentions_5m: u32,
    pub twitter_mentions_1h: u32,
    pub telegram_mentions_5m: u32,
    pub telegram_mentions_1h: u32,
    pub sentiment_score: f64,
    pub sentiment_acceleration: f64,
    pub kol_mention: bool,
    pub kol_names: Vec<String>,
    pub message_samples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeData {
    pub source_wallet: String,
    pub source_wallet_winrate: f64,
    pub source_wallet_total_trades: u32,
    pub buy_amount_sol: f64,
    pub buy_price_usd: f64,
    pub tx_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterResult {
    pub passed: bool,
    pub rejection_reason: Option<String>,
    pub liquidity_ok: bool,
    pub market_cap_ok: bool,
    pub token_age_ok: bool,
    pub dev_holding_ok: bool,
    pub sniper_ok: bool,
    pub mint_authority_ok: bool,
    pub freeze_authority_ok: bool,
    pub lp_lock_ok: bool,
    pub creator_history_ok: bool,
    pub price_impact_ok: bool,
    pub tabular_score: f64,
    pub transformer_score: f64,
    pub gnn_score: f64,
    pub nlp_score: f64,
    pub ensemble_score: f64,
    pub win_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeDecision {
    pub id: Uuid,
    pub signal: TokenSignal,
    pub filter_result: FilterResult,
    pub decision_type: DecisionType,
    pub strategy_track: StrategyTrack,
    pub buy_amount_sol: f64,
    pub max_slippage_bps: u32,
    pub entry_delay_seconds: u32,
    pub take_profit_1: f64,
    pub take_profit_2: f64,
    pub stop_loss: f64,
    pub time_stop_minutes: u32,
    pub decided_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecisionType { Buy, Skip, Sell, PartialSell { pct: f64 } }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StrategyTrack { Snipe, CopyTrade, Sentiment }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: Uuid,
    pub mint: String,
    pub strategy_track: StrategyTrack,
    pub status: TradeStatus,
    pub entry_price_usd: f64,
    pub entry_amount_sol: f64,
    pub entry_tx: Option<String>,
    pub entered_at: Option<DateTime<Utc>>,
    pub exit_price_usd: Option<f64>,
    pub exit_amount_sol: Option<f64>,
    pub exit_tx: Option<String>,
    pub exited_at: Option<DateTime<Utc>>,
    pub pnl_sol: Option<f64>,
    pub pnl_pct: Option<f64>,
    pub peak_multiplier: Option<f64>,
    pub jito_tip_lamports: Option<u64>,
    pub filter_result: FilterResult,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TradeStatus { Pending, Submitted, Confirmed, PartialExit, Closed, Failed, Cancelled }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionStats {
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub win_rate: f64,
    pub total_pnl_sol: f64,
    pub best_trade_pct: f64,
    pub worst_trade_pct: f64,
    pub avg_hold_minutes: f64,
    pub open_positions: u32,
    pub signals_scanned: u64,
    pub signals_filtered_out: u64,
    pub jito_tips_paid_sol: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RejectionStat {
    pub reason: String,
    pub count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingStats {
    pub total_tokens: i64,
    pub labeled_tokens: i64,
    pub positive_labels: i64,
    pub negative_labels: i64,
    pub hours_of_data: f64,
    pub collection_started_at: Option<DateTime<Utc>>,
    pub total_signals: i64,
    pub twitter_signals: i64,
    pub telegram_signals: i64,
    pub yellowstone_signals: i64,
    pub copy_trade_signals: i64,
    pub tokens_passed_filter: i64,
    pub tokens_per_hour: f64,
    pub filter_pass_rate: f64,
    pub top_rejections: Vec<RejectionStat>,
}
