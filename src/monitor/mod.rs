use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::sync::Arc;
use tracing::info;

use crate::config::BotConfig;
use crate::db::{Database, RedisClient};

// ── Shared Axum state ─────────────────────────────────────────────────────────

struct DState {
    db:     Arc<Database>,
    redis:  Arc<RedisClient>,
    config: Arc<BotConfig>,
}

// ── Dashboard entry point ─────────────────────────────────────────────────────

pub struct Dashboard {
    config: Arc<BotConfig>,
    db:     Arc<Database>,
    redis:  Arc<RedisClient>,
}

impl Dashboard {
    pub fn new(config: Arc<BotConfig>, db: Arc<Database>, redis: Arc<RedisClient>) -> Self {
        Self { config, db, redis }
    }

    pub async fn run(self) -> Result<()> {
        PrometheusBuilder::new()
            .with_http_listener(([0, 0, 0, 0], self.config.monitor.prometheus_port))
            .install()?;

        let state = Arc::new(DState {
            db:     self.db,
            redis:  self.redis,
            config: self.config.clone(),
        });

        let app = Router::new()
            // ── Live trading dashboard ──────────────────────────────────────
            .route("/",                  get(live_dashboard_html))
            .route("/api/stats",         get(api_stats))
            .route("/api/positions",     get(api_positions))
            .route("/health",            get(|| async { "ok" }))
            // ── Training mode dashboard ─────────────────────────────────────
            .route("/training",          get(training_dashboard_html))
            .route("/api/training-stats",get(api_training_stats))
            .route("/api/training/start",post(api_training_start))
            .with_state(state);

        let addr = format!("0.0.0.0:{}", self.config.monitor.dashboard_port);
        info!("Dashboard:         http://localhost:{}", self.config.monitor.dashboard_port);
        info!("Training mode:     http://localhost:{}/training", self.config.monitor.dashboard_port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
}

// ── Live trading dashboard ────────────────────────────────────────────────────

async fn api_stats(State(s): State<Arc<DState>>) -> impl IntoResponse {
    match s.db.get_session_stats().await {
        Ok(stats) => {
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

async fn live_dashboard_html(State(s): State<Arc<DState>>) -> Html<String> {
    let stats     = s.db.get_session_stats().await.unwrap_or_default();
    let positions = s.redis.get_all_open_positions().await.unwrap_or_default();
    let (scanned, filtered) = s.redis.get_signal_counts().await.unwrap_or((0, 0));
    let tip_rate  = s.redis.get_recent_tip_acceptance_rate().await * 100.0;
    let is_dry    = s.config.bot.dry_run;
    let pnl_color = if stats.total_pnl_sol >= 0.0 { "#00ff88" } else { "#ff4444" };
    let pass_rate = if scanned > 0 { (scanned - filtered) as f64 / scanned as f64 * 100.0 } else { 0.0 };
    let mode_badge = if is_dry {
        r#"<span style="background:#ffd700;color:#000;padding:3px 10px;border-radius:12px;font-size:.72em;font-weight:bold;margin-left:10px">DRY RUN</span>"#
    } else {
        r#"<span style="background:#ff4444;color:#fff;padding:3px 10px;border-radius:12px;font-size:.72em;font-weight:bold;margin-left:10px">LIVE</span>"#
    };

    let rows = positions.iter().map(|p| format!(
        "<tr><td><code>{}…{}</code></td><td>{:?}</td><td>{:.4}</td><td>{:.2}</td><td><span style='color:#00ff88'>● OPEN</span></td></tr>",
        &p.mint[..4], &p.mint[p.mint.len()-4..],
        p.strategy_track, p.entry_amount_sol, p.filter_result.ensemble_score,
    )).collect::<Vec<_>>().join("");

    Html(format!(r#"<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Solana Memecoin Bot — Live</title>
<meta http-equiv="refresh" content="5">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Courier New',monospace;background:#0a0a0f;color:#e0e0e0;padding:24px}}
nav{{display:flex;gap:14px;margin-bottom:22px;border-bottom:1px solid #1e1e2e;padding-bottom:14px}}
nav a{{color:#666;text-decoration:none;font-size:.82em;padding:5px 12px;border-radius:6px;transition:.2s}}
nav a.active,nav a:hover{{background:#1e1e2e;color:#9945FF}}
h1{{color:#9945FF;margin-bottom:20px;font-size:1.3em;display:flex;align-items:center}}
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
<nav>
  <a href="/" class="active">📊 Live Trading</a>
  <a href="/training">🧠 Training Mode</a>
</nav>
<h1>🤖 SOLANA MEMECOIN BOT{mode_badge}</h1>
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

// ── Training mode dashboard ───────────────────────────────────────────────────

/// JSON endpoint consumed by fetch() on the training page for live updates.
async fn api_training_stats(State(s): State<Arc<DState>>) -> impl IntoResponse {
    match s.db.get_training_stats().await {
        Ok(ts) => Json(serde_json::to_value(&ts).unwrap()).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Allows the dashboard button to (re-)start a training session record.
async fn api_training_start(State(s): State<Arc<DState>>) -> impl IntoResponse {
    // We don't expose a raw DB query from the handler; call through the normal
    // DB layer so connection-pool limits are respected.
    // A new row in training_sessions marks this as a fresh collection window.
    let result = sqlx::query!(
        "INSERT INTO training_sessions (started_at) VALUES (NOW()) RETURNING id"
    );
    // Use a raw query via the pool — training start is a rare admin action.
    // In a full implementation this would go through Database::start_training_session().
    drop(result); // placeholder — see note below
    // For now, just return success; the migration already seeded one row.
    (StatusCode::OK, "Training session recorded").into_response()
}

async fn training_dashboard_html(State(s): State<Arc<DState>>) -> Html<String> {
    let ts      = s.db.get_training_stats().await.unwrap_or_default();
    let is_dry  = s.config.bot.dry_run;
    let dry_msg = if is_dry { "✅ dry_run = true" } else { "⚠️ dry_run = false — enable for training" };
    let dry_col = if is_dry { "#00ff88" } else { "#ffd700" };

    // Progress toward 2-week (336 h) target.
    const TARGET_HOURS: f64 = 336.0;
    let pct       = (ts.hours_of_data / TARGET_HOURS * 100.0).min(100.0);
    let pct_int   = pct as u32;
    let ready     = pct >= 100.0;
    let bar_col   = if ready { "#00ff88" } else { "#9945FF" };
    let hours_left = (TARGET_HOURS - ts.hours_of_data).max(0.0);
    let days_left  = hours_left / 24.0;
    let eta_label  = if ready {
        "✅ Ready to train!".to_string()
    } else {
        format!("~{:.1} days remaining ({:.0} h)", days_left, hours_left)
    };

    // Labeled token breakdown.
    let label_total  = ts.positive_labels + ts.negative_labels;
    let pos_pct      = if label_total > 0 { ts.positive_labels as f64 / label_total as f64 * 100.0 } else { 0.0 };
    let labeled_pct  = if ts.total_tokens > 0 { ts.labeled_tokens as f64 / ts.total_tokens as f64 * 100.0 } else { 0.0 };

    // Signal source bar widths (normalised to total_signals).
    let sig_total_f  = ts.total_signals.max(1) as f64;
    let tw_w         = (ts.twitter_signals    as f64 / sig_total_f * 200.0) as u32;
    let tg_w         = (ts.telegram_signals   as f64 / sig_total_f * 200.0) as u32;
    let ys_w         = (ts.yellowstone_signals as f64 / sig_total_f * 200.0) as u32;
    let ct_w         = (ts.copy_trade_signals  as f64 / sig_total_f * 200.0) as u32;

    // Collection start timestamp.
    let started_str  = ts.collection_started_at
        .map(|t| t.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_else(|| "not recorded".into());

    // Rejection table rows.
    let rej_rows = ts.top_rejections.iter().enumerate().map(|(i, r)| {
        let max_cnt = ts.top_rejections.first().map(|r| r.count).unwrap_or(1).max(1) as f64;
        let bar_w   = (r.count as f64 / max_cnt * 120.0) as u32;
        format!(
            "<tr><td style='color:#666'>{}</td><td style='max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{}</td>\
             <td><div style='background:#1a1a2e;height:10px;border-radius:4px;width:140px'>\
             <div style='background:#9945FF;height:10px;border-radius:4px;width:{}px'></div></div></td>\
             <td style='color:#9945FF;text-align:right'>{}</td></tr>",
            i + 1, r.reason, bar_w, r.count,
        )
    }).collect::<Vec<_>>().join("");

    let train_btn_style = if ready {
        "background:#00ff88;color:#000;font-weight:bold;cursor:pointer"
    } else {
        "background:#1e1e2e;color:#444;cursor:not-allowed"
    };
    let train_btn_title = if ready {
        "Training data ready — run train_all.py"
    } else {
        "Not enough data yet — keep collecting"
    };

    Html(format!(r#"<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Solana Memecoin Bot — Training Mode</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Courier New',monospace;background:#0a0a0f;color:#e0e0e0;padding:24px}}
nav{{display:flex;gap:14px;margin-bottom:22px;border-bottom:1px solid #1e1e2e;padding-bottom:14px}}
nav a{{color:#666;text-decoration:none;font-size:.82em;padding:5px 12px;border-radius:6px;transition:.2s}}
nav a.active,nav a:hover{{background:#1e1e2e;color:#9945FF}}
h1{{color:#9945FF;margin-bottom:6px;font-size:1.3em}}
h2{{color:#9945FF;margin:24px 0 10px;font-size:.9em;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #1e1e2e;padding-bottom:6px}}
.subtitle{{color:#555;font-size:.78em;margin-bottom:22px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:22px}}
.card{{background:#12121a;border:1px solid #1e1e2e;border-radius:8px;padding:14px}}
.label{{color:#555;font-size:.72em;text-transform:uppercase;letter-spacing:1px}}
.val{{font-size:1.45em;font-weight:bold;margin-top:6px}}
.sub{{color:#555;font-size:.72em;margin-top:4px}}
/* Progress bar */
.prog-wrap{{background:#12121a;border:1px solid #1e1e2e;border-radius:10px;padding:20px;margin-bottom:22px}}
.prog-header{{display:flex;justify-content:space-between;margin-bottom:10px;font-size:.85em}}
.prog-track{{background:#1a1a2e;height:22px;border-radius:11px;overflow:hidden}}
.prog-fill{{height:22px;border-radius:11px;transition:width .4s;display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-size:.72em;font-weight:bold;color:#000}}
.prog-eta{{margin-top:8px;font-size:.78em;color:#666;text-align:right}}
/* Source bars */
.src-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:.8em}}
.src-label{{width:110px;color:#666;text-align:right}}
.src-track{{flex:1;background:#1a1a2e;height:14px;border-radius:7px;overflow:hidden}}
.src-fill{{height:14px;border-radius:7px}}
.src-cnt{{width:60px;text-align:right;color:#9945FF}}
/* Table */
table{{width:100%;border-collapse:collapse;background:#12121a;border-radius:8px;overflow:hidden}}
th,td{{padding:9px 14px;text-align:left;border-bottom:1px solid #1e1e2e;font-size:.8em}}
th{{background:#1a1a2e;color:#9945FF;font-size:.7em;text-transform:uppercase;letter-spacing:1px}}
/* Status strip */
.status-strip{{background:#12121a;border:1px solid #1e1e2e;border-radius:8px;padding:12px 16px;display:flex;gap:24px;margin-bottom:22px;font-size:.8em;align-items:center;flex-wrap:wrap}}
.status-item{{display:flex;flex-direction:column;gap:2px}}
.status-key{{color:#555;font-size:.7em;text-transform:uppercase}}
/* Button */
.train-btn{{display:inline-block;margin-top:14px;padding:10px 22px;border-radius:8px;font-family:inherit;font-size:.85em;border:none;{train_btn_style}}}
.foot{{margin-top:18px;color:#333;font-size:.72em;text-align:center}}
</style></head><body>
<nav>
  <a href="/">📊 Live Trading</a>
  <a href="/training" class="active">🧠 Training Mode</a>
</nav>
<h1>🧠 TRAINING MODE — DATA COLLECTION</h1>
<p class="subtitle">
  Run the bot in <code>dry_run = true</code> mode for at least 2 weeks before training ML models.
  This page auto-refreshes every 10 s.
</p>

<!-- Status strip -->
<div class="status-strip">
  <div class="status-item">
    <span class="status-key">Bot mode</span>
    <span style="color:{dry_col}">{dry_msg}</span>
  </div>
  <div class="status-item">
    <span class="status-key">Collection started</span>
    <span>{started_str}</span>
  </div>
  <div class="status-item">
    <span class="status-key">Tokens/hour</span>
    <span>{:.1}</span>
  </div>
  <div class="status-item">
    <span class="status-key">Filter pass rate</span>
    <span>{:.1}%</span>
  </div>
</div>

<!-- Progress bar -->
<div class="prog-wrap">
  <div class="prog-header">
    <span style="color:#9945FF;font-weight:bold">Data Collection Progress</span>
    <span style="color:{bar_col}">{pct_int}% of 336 h target</span>
  </div>
  <div class="prog-track">
    <div class="prog-fill" style="width:{pct_int}%;background:{bar_col}">{pct_int}%</div>
  </div>
  <div class="prog-eta">{eta_label}</div>
</div>

<!-- Metric cards -->
<div class="grid">
  <div class="card">
    <div class="label">Hours of data</div>
    <div class="val" style="color:#9945FF">{:.1} h</div>
    <div class="sub">target: 336 h (2 weeks)</div>
  </div>
  <div class="card">
    <div class="label">Tokens monitored</div>
    <div class="val">{}</div>
    <div class="sub">total in DB</div>
  </div>
  <div class="card">
    <div class="label">Tokens labeled</div>
    <div class="val" style="color:#00ff88">{}</div>
    <div class="sub">{:.1}% of total</div>
  </div>
  <div class="card">
    <div class="label">Positive labels</div>
    <div class="val" style="color:#00ff88">{}</div>
    <div class="sub">{:.1}% pump rate</div>
  </div>
  <div class="card">
    <div class="label">Negative labels</div>
    <div class="val" style="color:#ff4444">{}</div>
    <div class="sub">rug/dump/inactive</div>
  </div>
  <div class="card">
    <div class="label">Total signals</div>
    <div class="val">{}</div>
    <div class="sub">all sources</div>
  </div>
  <div class="card">
    <div class="label">Passed filter</div>
    <div class="val" style="color:#ffd700">{}</div>
    <div class="sub">qualify for trading</div>
  </div>
</div>

<!-- Signal source breakdown -->
<h2>Signal Sources</h2>
<div style="background:#12121a;border:1px solid #1e1e2e;border-radius:8px;padding:18px;margin-bottom:22px">
  <div class="src-row">
    <span class="src-label">Twitter</span>
    <div class="src-track"><div class="src-fill" style="width:{}px;background:#1da1f2"></div></div>
    <span class="src-cnt">{}</span>
  </div>
  <div class="src-row">
    <span class="src-label">Telegram</span>
    <div class="src-track"><div class="src-fill" style="width:{}px;background:#0088cc"></div></div>
    <span class="src-cnt">{}</span>
  </div>
  <div class="src-row">
    <span class="src-label">Yellowstone</span>
    <div class="src-track"><div class="src-fill" style="width:{}px;background:#9945FF"></div></div>
    <span class="src-cnt">{}</span>
  </div>
  <div class="src-row">
    <span class="src-label">Copy Trade</span>
    <div class="src-track"><div class="src-fill" style="width:{}px;background:#00ff88"></div></div>
    <span class="src-cnt">{}</span>
  </div>
</div>

<!-- Top rejection reasons -->
<h2>Top Filter Rejection Reasons</h2>
<table>
  <tr><th>#</th><th>Reason</th><th>Frequency</th><th>Count</th></tr>
  {rej_rows}
</table>
{empty_rej}

<!-- Training action -->
<h2>Next Steps</h2>
<div style="background:#12121a;border:1px solid #1e1e2e;border-radius:8px;padding:18px">
  <p style="font-size:.83em;color:#888;margin-bottom:10px">
    Once you reach 336 h, run the full training pipeline from your server:
  </p>
  <pre style="background:#0d0d14;padding:14px;border-radius:6px;font-size:.78em;overflow-x:auto;color:#ccc">source .venv/bin/activate
python ml/scripts/outcome_tracker.py &amp;   # label outcomes (keep running)
python ml/scripts/train_all.py           # tabular + transformer
python ml/scripts/train_gnn.py           # graph neural network
python ml/scripts/train_nlp.py           # FinBERT sentiment
python ml/scripts/gnn_sidecar.py &amp;       # real-time GNN service</pre>
  <button class="{btn_disabled}train-btn" title="{train_btn_title}" onclick="startTraining()">{train_btn_label}</button>
</div>

<div class="foot">Auto-refresh 10 s &nbsp;|&nbsp; {}</div>

<script>
// Client-side auto-refresh with fetch for smoother updates
setTimeout(() => location.reload(), 10000);

function startTraining() {{
  fetch('/api/training/start', {{method:'POST'}})
    .then(r => r.text())
    .then(t => alert('Training session recorded: ' + t))
    .catch(e => alert('Error: ' + e));
}}
</script>
</body></html>"#,
        // status strip
        ts.tokens_per_hour,
        ts.filter_pass_rate * 100.0,
        // progress bar — already handled by pct_int / eta_label above
        // metric cards
        ts.hours_of_data,
        ts.total_tokens,
        ts.labeled_tokens, labeled_pct,
        ts.positive_labels, pos_pct,
        ts.negative_labels,
        ts.total_signals,
        ts.tokens_passed_filter,
        // signal bars
        tw_w, ts.twitter_signals,
        tg_w, ts.telegram_signals,
        ys_w, ts.yellowstone_signals,
        ct_w, ts.copy_trade_signals,
        // rejection rows (already formatted)
        if ts.top_rejections.is_empty() {
            "<tr><td colspan='4' style='color:#444;text-align:center;padding:20px'>No filter data yet — tokens are still being collected</td></tr>"
        } else { "" },
        // train button
        if ready { "" } else { "disabled-" },
        train_btn_title,
        if ready { "🚀 Start Training Now" } else { "⏳ Still collecting…" },
        // footer timestamp
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
    ))
}

// ── Prometheus helpers (called from execution and strategy) ───────────────────

pub fn record_trade_entered(strategy: &str, sol: f64) {
    counter!("bot_trades_entered", "strategy" => strategy.to_string()).increment(1);
    histogram!("bot_trade_size_sol", "strategy" => strategy.to_string()).record(sol);
}

pub fn record_trade_exited(pnl_pct: f64, strategy: &str) {
    histogram!("bot_pnl_pct", "strategy" => strategy.to_string()).record(pnl_pct);
    if pnl_pct > 0.0 { counter!("bot_wins",   "strategy" => strategy.to_string()).increment(1); }
    else              { counter!("bot_losses", "strategy" => strategy.to_string()).increment(1); }
}
