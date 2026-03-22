use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BotConfig {
    pub bot: BotSettings, pub wallet: WalletSettings, pub rpc: RpcSettings,
    pub jito: JitoSettings, pub bloxroute: BloxrouteSettings, pub nozomi: NozomiSettings,
    pub database: DatabaseSettings, pub signals: SignalSettings, pub filter: FilterSettings,
    pub strategy: StrategySettings, pub execution: ExecutionSettings,
    pub models: ModelsConfig, pub monitor: MonitorSettings,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BotSettings {
    pub name: String, pub dry_run: bool, pub log_level: String,
    pub max_trades_per_hour: u32,
    pub trading_window_start_utc: u32, pub trading_window_end_utc: u32,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WalletSettings {
    pub keypair_path: String, pub max_total_capital_sol: f64,
    pub max_position_size_sol: f64, pub max_concurrent_positions: u32, pub reserve_sol: f64,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RpcSettings {
    pub helius_api_key: String, pub helius_rpc_url: String, pub helius_ws_url: String,
    pub yellowstone_grpc_url: String, pub yellowstone_grpc_token: String,
    pub quicknode_url: String, pub quicknode_ws_url: String, pub shyft_api_key: String,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JitoSettings {
    pub block_engines: Vec<String>, pub auth_keypair_path: String,
    pub tip_percentile_target: u32, pub tip_min_lamports: u64,
    pub tip_max_lamports: u64, pub tip_profit_share: f64, pub tip_account: String,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BloxrouteSettings { pub enabled: bool, pub auth_header: String, pub endpoint: String }
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NozomiSettings { pub enabled: bool, pub api_key: String, pub endpoint: String }
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatabaseSettings {
    pub postgres_url: String, pub redis_url: String, pub pool_max_connections: u32,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SignalSettings { pub twitter: TwitterSettings, pub telegram: TelegramSettings }
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TwitterSettings {
    pub bearer_token: String, pub kol_accounts: Vec<String>,
    pub trigger_keywords: Vec<String>, pub min_kol_followers: u32,
    pub sentiment_score_threshold: f64,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TelegramSettings {
    pub api_id: String, pub api_hash: String, pub phone: String,
    pub groups: Vec<String>, pub min_message_velocity: u32,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FilterSettings {
    pub min_liquidity_usd: f64, pub max_liquidity_usd: f64,
    pub min_market_cap_usd: f64, pub max_market_cap_usd: f64,
    pub max_token_age_hours: u32, pub min_buy_count_24h: u32, pub min_sell_count_24h: u32,
    pub max_dev_holding_pct: f64, pub max_sniper_concentration_pct: f64,
    pub min_win_probability: f64, pub min_sentiment_score: f64,
    pub require_mint_authority_disabled: bool, pub require_freeze_authority_disabled: bool,
    pub min_lp_lock_days: u32, pub max_creator_rug_history: u32,
    pub max_price_impact_5pct_sell: f64,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StrategySettings {
    pub snipe_entry_1_pct: f64, pub snipe_entry_2_delay_secs: u64, pub snipe_entry_2_pct: f64,
    pub snipe_entry_3_delay_secs: u64, pub snipe_entry_3_pct: f64,
    pub tp_1_multiplier: f64, pub tp_1_sell_pct: f64,
    pub tp_2_multiplier: f64, pub tp_2_sell_pct: f64, pub tp_3_trailing_stop_pct: f64,
    pub hard_stop_loss_pct: f64, pub time_stop_minutes: u32, pub max_daily_loss_sol: f64,
    pub circuit_breaker_consecutive_losses: u32, pub circuit_breaker_pause_minutes: u32,
    pub copy_trade_enabled: bool, pub copy_wallets: Vec<String>,
    pub copy_max_buy_sol: f64, pub copy_wallet_min_winrate: f64,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExecutionSettings {
    pub skip_preflight: bool, pub commitment: String, pub max_retries: u32,
    pub retry_delay_ms: u64, pub slippage_start_bps: u32,
    pub slippage_max_bps: u32, pub slippage_increment_bps: u32,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsConfig {
    pub tabular_model_path: String, pub transformer_model_path: String,
    pub gnn_model_path: String, pub nlp_model_path: String,
    pub tabular_weight: f64, pub transformer_weight: f64,
    pub gnn_weight: f64, pub nlp_weight: f64,
    pub ensemble_threshold: f64, pub retrain_cron: String,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MonitorSettings {
    pub prometheus_port: u16, pub dashboard_port: u16, pub alert_webhook_url: String,
    pub alert_on_trade: bool, pub alert_on_loss_pct: f64, pub alert_on_circuit_breaker: bool,
}

impl BotConfig {
    pub fn load() -> Result<Self> {
        let cfg = config::Config::builder()
            .add_source(config::File::with_name("config/default").required(true))
            .add_source(config::File::with_name("config/local").required(false))
            .add_source(config::Environment::with_prefix("BOTCONFIG").separator("__"))
            .build()?;
        Ok(cfg.try_deserialize()?)
    }
}
