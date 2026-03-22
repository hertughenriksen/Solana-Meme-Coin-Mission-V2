use anyhow::Result;
use chrono::{Datelike, Timelike};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod db;
mod scanner;
mod signals;
mod filter;
mod strategy;
mod execution;
mod models;
mod monitor;
mod price_feed;
mod types;

use config::BotConfig;
use db::{Database, RedisClient};
use scanner::YellowstoneScanner;
use signals::{TwitterScanner, TelegramScanner, WalletTracker};
use filter::FilterPipeline;
use strategy::StrategyEngine;
use execution::ExecutionEngine;
use models::ModelEnsemble;
use monitor::Dashboard;
use price_feed::PriceFeed;
use types::{TokenSignal, TradeDecision};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,solana_memecoin_bot=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .init();

    info!("╔══════════════════════════════════════════╗");
    info!("║     SOLANA MEMECOIN BOT v0.1.0            ║");
    info!("║     Starting all systems...               ║");
    info!("╚══════════════════════════════════════════╝");

    let config = Arc::new(BotConfig::load()?);
    info!("Config loaded | dry_run={}", config.bot.dry_run);
    if config.bot.dry_run {
        warn!("⚠️  DRY RUN / TRAINING MODE — no real transactions will be sent");
        info!("📊 Training dashboard: http://localhost:{}/training", config.monitor.dashboard_port);
    }

    info!("Connecting to PostgreSQL...");
    let db = Arc::new(Database::connect(&config.database.postgres_url).await?);
    db.run_migrations().await?;

    info!("Connecting to Redis...");
    let redis = Arc::new(RedisClient::connect(&config.database.redis_url).await?);

    info!("Loading ONNX model ensemble...");
    let models = Arc::new(ModelEnsemble::load(&config.models)?);

    let (signal_tx, _signal_rx) = broadcast::channel::<TokenSignal>(1024);
    let (trade_tx, trade_rx)    = mpsc::channel::<TradeDecision>(256);

    let filter = Arc::new(FilterPipeline::new(
        config.clone(), db.clone(), models.clone(), redis.clone(),
    ));
    let strategy = Arc::new(StrategyEngine::new(
        config.clone(), db.clone(), redis.clone(), filter.clone(), trade_tx,
    ));
    let execution = Arc::new(
        ExecutionEngine::new(config.clone(), db.clone(), redis.clone()).await?,
    );
    let dashboard = Dashboard::new(config.clone(), db.clone(), redis.clone());

    // ── Price feed ────────────────────────────────────────────────────────
    let pf = PriceFeed::new(redis.clone());
    tokio::spawn(async move { pf.run().await; });
    info!("✅ Price feed started");

    // ── Signal sources ────────────────────────────────────────────────────
    let ys = YellowstoneScanner::new(config.clone(), signal_tx.clone());
    tokio::spawn(async move {
        loop {
            if let Err(e) = ys.run().await {
                error!("Yellowstone error: {e} — reconnecting in 5s");
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }
    });

    let tw = TwitterScanner::new(config.clone(), signal_tx.clone());
    tokio::spawn(async move {
        loop {
            if let Err(e) = tw.run().await {
                error!("Twitter scanner error: {e} — reconnecting in 15s");
                tokio::time::sleep(tokio::time::Duration::from_secs(15)).await;
            }
        }
    });

    let tg = TelegramScanner::new(config.clone(), signal_tx.clone());
    tokio::spawn(async move {
        loop {
            if let Err(e) = tg.run().await {
                error!("Telegram scanner error: {e} — reconnecting in 10s");
                tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            }
        }
    });

    if config.strategy.copy_trade_enabled && !config.strategy.copy_wallets.is_empty() {
        let wt = WalletTracker::new(config.clone(), signal_tx.clone());
        tokio::spawn(async move {
            loop {
                if let Err(e) = wt.run().await {
                    error!("Wallet tracker error: {e} — reconnecting in 5s");
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        });
    }

    // ── Strategy engine ───────────────────────────────────────────────────
    let strat = strategy.clone();
    let mut sig_rx = signal_tx.subscribe();
    tokio::spawn(async move {
        while let Ok(signal) = sig_rx.recv().await {
            let s = strat.clone();
            tokio::spawn(async move {
                if let Err(e) = s.process_signal(signal).await {
                    error!("Strategy error: {e}");
                }
            });
        }
    });

    // ── Execution engine ──────────────────────────────────────────────────
    let exec = execution.clone();
    tokio::spawn(async move { exec.run(trade_rx).await });

    // ── Position manager ──────────────────────────────────────────────────
    let strat_pm = strategy.clone();
    tokio::spawn(async move {
        let mut iv = tokio::time::interval(tokio::time::Duration::from_secs(2));
        loop {
            iv.tick().await;
            if let Err(e) = strat_pm.manage_positions().await {
                error!("Position manager error: {e}");
            }
        }
    });

    // ── Weekly retraining ─────────────────────────────────────────────────
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
            let now = chrono::Utc::now();
            if now.weekday() == chrono::Weekday::Sun && now.hour() == 2 {
                info!("🔄 Starting scheduled model retraining...");
                if let Err(e) = models::retrain_models().await {
                    error!("Model retraining failed: {e}");
                }
            }
        }
    });

    // ── Dashboard ─────────────────────────────────────────────────────────
    info!("Starting dashboard on port {}...", config.monitor.dashboard_port);
    tokio::spawn(async move {
        if let Err(e) = dashboard.run().await {
            error!("Dashboard error: {e}");
        }
    });

    info!("✅ All systems online");
    info!("📊 Live dashboard:     http://localhost:{}", config.monitor.dashboard_port);
    info!("🧠 Training dashboard: http://localhost:{}/training", config.monitor.dashboard_port);

    let db_stats    = db.clone();
    let redis_stats = redis.clone();
    let mut iv = tokio::time::interval(tokio::time::Duration::from_secs(60));
    loop {
        iv.tick().await;
        match db_stats.get_session_stats().await {
            Ok(stats) => {
                let (scanned, filtered) = redis_stats.get_signal_counts().await.unwrap_or((0, 0));
                info!(
                    "📊 trades={} win={:.1}% pnl={:+.4}SOL open={} scanned={} pass_rate={:.1}%",
                    stats.total_trades, stats.win_rate * 100.0, stats.total_pnl_sol,
                    stats.open_positions, scanned,
                    if scanned > 0 { (scanned - filtered) as f64 / scanned as f64 * 100.0 } else { 0.0 },
                );
            }
            Err(e) => error!("Stats error: {e}"),
        }
    }
}
