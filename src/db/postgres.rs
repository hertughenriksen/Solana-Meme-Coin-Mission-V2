use anyhow::Result;
use sqlx::{PgPool, postgres::PgPoolOptions, Row};
use tracing::info;
use uuid::Uuid;
use crate::types::*;

pub struct Database { pool: PgPool }

impl Database {
    pub async fn connect(url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(20).min_connections(2)
            .acquire_timeout(std::time::Duration::from_secs(10))
            .connect(url).await?;
        info!("PostgreSQL connected");
        Ok(Self { pool })
    }

    pub async fn run_migrations(&self) -> Result<()> {
        sqlx::migrate!("./migrations").run(&self.pool).await?;
        info!("Migrations applied");
        Ok(())
    }

    pub async fn insert_trade(&self, trade: &Trade) -> Result<()> {
        sqlx::query(
            r#"INSERT INTO trades (
                id, mint, strategy_track, status, entry_price_usd, entry_amount_sol,
                ml_tabular_score, ml_transformer_score, ml_gnn_score, ml_nlp_score,
                ml_ensemble_score, jito_tip_lamports, created_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (id) DO NOTHING"#,
        )
        .bind(trade.id)
        .bind(&trade.mint)
        .bind(format!("{:?}", trade.strategy_track).to_lowercase())
        .bind(format!("{:?}", trade.status).to_lowercase())
        .bind(trade.entry_price_usd)
        .bind(trade.entry_amount_sol)
        .bind(trade.filter_result.tabular_score)
        .bind(trade.filter_result.transformer_score)
        .bind(trade.filter_result.gnn_score)
        .bind(trade.filter_result.nlp_score)
        .bind(trade.filter_result.ensemble_score)
        .bind(trade.jito_tip_lamports.map(|t| t as i64))
        .bind(trade.created_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn update_trade_status(&self, id: Uuid, status: TradeStatus) -> Result<()> {
        sqlx::query("UPDATE trades SET status = $1 WHERE id = $2")
            .bind(format!("{:?}", status).to_lowercase())
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn log_dry_run_trade(&self, decision: &TradeDecision) -> Result<()> {
        sqlx::query(
            r#"INSERT INTO trades (id, mint, strategy_track, status, entry_amount_sol, ml_ensemble_score, created_at)
               VALUES ($1,$2,$3,'cancelled',$4,$5,NOW()) ON CONFLICT DO NOTHING"#,
        )
        .bind(decision.id)
        .bind(&decision.signal.mint)
        .bind(format!("{:?}", decision.strategy_track).to_lowercase())
        .bind(decision.buy_amount_sol)
        .bind(decision.filter_result.ensemble_score)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn log_filter_result(&self, mint: &str, result: &FilterResult) -> Result<()> {
        sqlx::query(
            r#"INSERT INTO tokens (mint, filter_passed, filter_rejection_reason, ml_ensemble_score)
               VALUES ($1,$2,$3,$4)
               ON CONFLICT (mint) DO UPDATE SET
                 filter_passed           = EXCLUDED.filter_passed,
                 filter_rejection_reason = EXCLUDED.filter_rejection_reason,
                 ml_ensemble_score       = EXCLUDED.ml_ensemble_score,
                 updated_at              = NOW()"#,
        )
        .bind(mint)
        .bind(result.passed)
        .bind(result.rejection_reason.as_deref())
        .bind(result.ensemble_score)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_deployer_rug_count(&self, wallet: &str) -> Result<Option<u32>> {
        let row = sqlx::query("SELECT rug_count FROM deployer_wallets WHERE wallet = $1")
            .bind(wallet)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.try_get::<i32, _>("rug_count").unwrap_or(0) as u32))
    }

    pub async fn get_copy_wallet_winrate(&self, wallet: &str) -> Result<Option<f64>> {
        let row = sqlx::query(
            "SELECT win_rate FROM copy_wallets WHERE wallet = $1 AND is_active = true",
        )
        .bind(wallet)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.and_then(|r| r.try_get::<f64, _>("win_rate").ok()))
    }

    pub async fn get_session_stats(&self) -> Result<SessionStats> {
        let row = sqlx::query(
            r#"SELECT
                COUNT(*)::BIGINT                                                             AS total_trades,
                SUM(CASE WHEN pnl_sol > 0  THEN 1 ELSE 0 END)::BIGINT                      AS winning_trades,
                SUM(CASE WHEN pnl_sol <= 0 AND status='closed' THEN 1 ELSE 0 END)::BIGINT  AS losing_trades,
                COALESCE(SUM(pnl_sol)::FLOAT8,  0.0)                                        AS total_pnl_sol,
                COALESCE(MAX(pnl_pct)::FLOAT8,  0.0)                                        AS best_trade_pct,
                COALESCE(MIN(pnl_pct)::FLOAT8,  0.0)                                        AS worst_trade_pct,
                COALESCE(AVG(EXTRACT(EPOCH FROM (exited_at - entered_at))/60)::FLOAT8, 0.0) AS avg_hold_minutes,
                COUNT(*) FILTER (WHERE status IN ('confirmed','partial_exit'))::BIGINT       AS open_positions,
                COALESCE(SUM(jito_tip_lamports)::BIGINT, 0)                                 AS jito_tips
            FROM trades
            WHERE created_at >= NOW() - INTERVAL '24 hours' AND status != 'cancelled'"#,
        )
        .fetch_one(&self.pool)
        .await?;

        let total = row.try_get::<i64, _>("total_trades").unwrap_or(0) as u32;
        let wins  = row.try_get::<i64, _>("winning_trades").unwrap_or(0) as u32;
        Ok(SessionStats {
            total_trades:         total,
            winning_trades:       wins,
            losing_trades:        row.try_get::<i64, _>("losing_trades").unwrap_or(0) as u32,
            win_rate:             if total > 0 { wins as f64 / total as f64 } else { 0.0 },
            total_pnl_sol:        row.try_get::<f64, _>("total_pnl_sol").unwrap_or(0.0),
            best_trade_pct:       row.try_get::<f64, _>("best_trade_pct").unwrap_or(0.0),
            worst_trade_pct:      row.try_get::<f64, _>("worst_trade_pct").unwrap_or(0.0),
            avg_hold_minutes:     row.try_get::<f64, _>("avg_hold_minutes").unwrap_or(0.0),
            open_positions:       row.try_get::<i64, _>("open_positions").unwrap_or(0) as u32,
            signals_scanned:      0,
            signals_filtered_out: 0,
            jito_tips_paid_sol:   row.try_get::<i64, _>("jito_tips").unwrap_or(0) as f64 / 1e9,
        })
    }

    pub async fn count_recent_losses(&self, hours: u32) -> Result<i64> {
        let row = sqlx::query(
            "SELECT COUNT(*) AS cnt FROM trades WHERE status='closed' AND pnl_sol < 0
             AND created_at >= NOW() - MAKE_INTERVAL(hours => $1)",
        )
        .bind(hours as i32)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.try_get::<i64, _>("cnt").unwrap_or(0))
    }

    pub async fn log_signal(&self, signal: &TokenSignal, passed: bool, rejection: Option<&str>) -> Result<()> {
        sqlx::query(
            r#"INSERT INTO signals (id, mint, source, signal_type, detected_at, filter_passed, filter_rejection)
               VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT DO NOTHING"#,
        )
        .bind(signal.id)
        .bind(&signal.mint)
        .bind(format!("{:?}", signal.source).to_lowercase())
        .bind(format!("{:?}", signal.signal_type).to_lowercase())
        .bind(signal.detected_at)
        .bind(passed)
        .bind(rejection)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn insert_price_candle(&self, mint: &str, candle: &Candle) -> Result<()> {
        sqlx::query(
            r#"INSERT INTO price_candles_2s (mint, ts, open, high, low, close, volume, buy_count, sell_count)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) ON CONFLICT DO NOTHING"#,
        )
        .bind(mint)
        .bind(candle.timestamp)
        .bind(candle.open)
        .bind(candle.high)
        .bind(candle.low)
        .bind(candle.close)
        .bind(candle.volume)
        .bind(candle.buy_count as i32)
        .bind(candle.sell_count as i32)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
