use anyhow::Result;
use axum::{
    extract::State, http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::get, Router,
};
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::sync::Arc;
use tracing::info;

use crate::config::BotConfig;
use crate::db::{Database, RedisClient};

pub struct Dashboard {
    config: Arc<BotConfig>,
    db: Arc<Database>,
    redis: Arc<RedisClient>,
}

impl Dashboard {
    pub fn new(config: Arc<BotConfig>, db: Arc<Database>, redis: Arc<RedisClient>) -> Self {
        Self { config, db, redis }
    }

    pub async fn run(self) -> Result<()> {
        PrometheusBuilder::new()
            .with_http_listener(([0, 0, 0, 0], self.config.monitor.prometheus_port))
            .install()?;

        let state = Arc::new(DState { db: self.db, redis: self.redis });

        let app = Router::new()
            .route("/",               get(dashboard_html))
            .route("/api/stats",      get(api_stats))
            .route("/api/positions",  get(api_positions))
            .route("/health",         get(|| async { "ok" }))
            .with_state(state);

        let addr = format!("0.0.0.0:{}", self.config.monitor.dashboard_port);
        info!("Dashboard: http://localhost:{}", self.config.monitor.dashboard_port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
}

struct DState { db: Arc<Database>, redis: Arc<RedisClient> }

async fn api_stats(State(s): State<Arc<DState>>) -> impl IntoResponse {
    match s.db.get_session_stats().await {
        Ok(stats) => {
            // metrics 0.22: macros return handle objects; call .set()/.increment()/.record()
            gauge!("bot_win_rate").set(stats.win_rate);
            gauge!("bot_pnl_sol").set(stats.total_pnl_sol);
            gauge!("bot_open_positions").set(stats.open_positions as f64);
            Json(serde_json::to_value(&stats).unwrap()).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn api_positions(State(s): State<Arc<DState>>) -> impl IntoResponse {
    match s.redis.get_all_open_positions().await {
        Ok(p)  => Json(serde_json::to_value(p).unwrap()).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn dashboard_html(State(s): State<Arc<DState>>) -> Html<String> {
    let stats     = s.db.get_session_stats().await.unwrap_or_default();
    let positions = s.redis.get_all_open_positions().await.unwrap_or_default();
    let (scanned, filtered) = s.redis.get_signal_counts().await.unwrap_or((0, 0));
    let tip_rate  = s.redis.get_recent_tip_acceptance_rate().await * 100.0;
    let pnl_color = if stats.total_pnl_sol >= 0.0 { "#00ff88" } else { "#ff4444" };
    let pass_rate = if scanned > 0 { (scanned - filtered) as f64 / scanned as f64 * 100.0 } else { 0.0 };

    let rows = positions.iter().map(|p| format!(
        "<tr><td><code>{}…{}</code></td><td>{:?}</td><td>{:.4}</td><td>{:.2}</td><td><span style='color:#00ff88'>● OPEN</span></td></tr>",
        &p.mint[..4], &p.mint[p.mint.len()-4..],
        p.strategy_track, p.entry_amount_sol, p.filter_result.ensemble_score,
    )).collect::<Vec<_>>().join("");

    Html(format!(r#"<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Solana Memecoin Bot</title>
<meta http-equiv="refresh" content="5">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Courier New',monospace;background:#0a0a0f;color:#e0e0e0;padding:24px}}
h1{{color:#9945FF;margin-bottom:20px;font-size:1.3em}}
h2{{color:#9945FF;margin:20px 0 10px;font-size:.95em;text-transform:uppercase;letter-spacing:1px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:24px}}
.card{{background:#12121a;border:1px solid #1e1e2e;border-radius:8px;padding:14px}}
.label{{color:#666;font-size:.75em;text-transform:uppercase;letter-spacing:1px}}
.val{{font-size:1.5em;font-weight:bold;margin-top:5px}}
table{{width:100%;border-collapse:collapse;background:#12121a;border-radius:8px;overflow:hidden}}
th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid #1e1e2e;font-size:.83em}}
th{{background:#1a1a2e;color:#9945FF;font-size:.72em;text-transform:uppercase;letter-spacing:1px}}
.foot{{margin-top:16px;color:#444;font-size:.72em;text-align:center}}
</style></head><body>
<h1>🤖 SOLANA MEMECOIN BOT — LIVE</h1>
<div class="grid">
  <div class="card"><div class="label">PnL (24h)</div><div class="val" style="color:{pnl_color}">{:+.4} SOL</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="val" style="color:#00ff88">{:.1}%</div></div>
  <div class="card"><div class="label">Trades</div><div class="val" style="color:#9945FF">{}</div></div>
  <div class="card"><div class="label">Open</div><div class="val" style="color:#ffd700">{}</div></div>
  <div class="card"><div class="label">Scanned</div><div class="val">{}</div></div>
  <div class="card"><div class="label">Filter Pass</div><div class="val">{:.1}%</div></div>
  <div class="card"><div class="label">Jito Tips</div><div class="val" style="color:#ff4444">{:.4} SOL</div></div>
  <div class="card"><div class="label">Bundle Accept</div><div class="val" style="color:#00ff88">{:.0}%</div></div>
</div>
<h2>Open Positions</h2>
<table><tr><th>Mint</th><th>Track</th><th>Size (SOL)</th><th>ML Score</th><th>Status</th></tr>
{rows}</table>
<div class="foot">Auto-refresh 5s | {}</div>
</body></html>"#,
        stats.total_pnl_sol, stats.win_rate * 100.0, stats.total_trades,
        stats.open_positions, scanned, pass_rate,
        stats.jito_tips_paid_sol, tip_rate,
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
    ))
}

/// metrics 0.22: counter/histogram labels come before the method call, value goes in .increment()/.record()
pub fn record_trade_entered(strategy: &str, sol: f64) {
    counter!("bot_trades_entered", "strategy" => strategy.to_string()).increment(1);
    histogram!("bot_trade_size_sol", "strategy" => strategy.to_string()).record(sol);
}

pub fn record_trade_exited(pnl_pct: f64, strategy: &str) {
    histogram!("bot_pnl_pct", "strategy" => strategy.to_string()).record(pnl_pct);
    if pnl_pct > 0.0 { counter!("bot_wins",   "strategy" => strategy.to_string()).increment(1); }
    else              { counter!("bot_losses", "strategy" => strategy.to_string()).increment(1); }
}
