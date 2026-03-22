use anyhow::Result;
use sqlx::{PgPool, postgres::PgPoolOptions};
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
        sqlx::query!(
            r#"INSERT INTO trades (
                id, mint, strategy_track, status, entry_price_usd, entry_amount_sol,
                ml_tabular_score, ml_transformer_score, ml_gnn_score, ml_nlp_score,
                ml_ensemble_score, jito_tip_lamports, created_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (id) DO NOTHING"#,
            trade.id, trade.mint,
            format!("{:?}", trade.strategy_track).to_lowercase(),
            format!("{:?}", trade.status).to_lowercase(),
            trade.entry_price_usd, trade.entry_amount_sol,
            trade.filter_result.tabular_score,
            trade.filter_result.transformer_score,
            trade.filter_result.gnn_score,
            trade.filter_result.nlp_score,
            trade.filter_result.ensemble_score,
            trade.jito_tip_lamports.map(|t| t as i64),
            trade.created_at,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn update_trade_status(&self, id: Uuid, status: TradeStatus) -> Result<()> {
        sqlx::query!(
            "UPDATE trades SET status = $1 WHERE id = $2",
            format!("{:?}", status).to_lowercase(), id,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn log_dry_run_trade(&self, decision: &TradeDecision) -> Result<()> {
        sqlx::query!(
            r#"INSERT INTO trades (id, mint, strategy_track, status, entry_amount_sol, ml_ensemble_score, created_at)
               VALUES ($1,$2,$3,'cancelled',$4,$5,NOW()) ON CONFLICT DO NOTHING"#,
            decision.id, decision.signal.mint,
            format!("{:?}", decision.strategy_track).to_lowercase(),
            decision.buy_amount_sol, decision.filter_result.ensemble_score,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn log_filter_result(&self, mint: &str, result: &FilterResult) -> Result<()> {
        sqlx::query!(
            r#"INSERT INTO tokens (mint, filter_passed, filter_rejection_reason, ml_ensemble_score)
               VALUES ($1,$2,$3,$4)
               ON CONFLICT (mint) DO UPDATE SET
                 filter_passed = EXCLUDED.filter_passed,
                 filter_rejection_reason = EXCLUDED.filter_rejection_reason,
                 ml_ensemble_score = EXCLUDED.ml_ensemble_score,
                 updated_at = NOW()"#,
            mint, result.passed, result.rejection_reason.as_deref(), result.ensemble_score,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn get_deployer_rug_count(&self, wallet: &str) -> Result<Option<u32>> {
        let row = sqlx::query!(
            "SELECT rug_count FROM deployer_wallets WHERE wallet = $1", wallet
        ).fetch_optional(&self.pool).await?;
        Ok(row.map(|r| r.rug_count as u32))
    }

    pub async fn get_copy_wallet_winrate(&self, wallet: &str) -> Result<Option<f64>> {
        let row = sqlx::query!(
            "SELECT win_rate FROM copy_wallets WHERE wallet = $1 AND is_active = true", wallet
        ).fetch_optional(&self.pool).await?;
        Ok(row.and_then(|r| r.win_rate))
    }

    pub async fn get_session_stats(&self) -> Result<SessionStats> {
        let row = sqlx::query!(r#"
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl_sol <= 0 AND status='closed' THEN 1 ELSE 0 END) as losing_trades,
                COALESCE(SUM(pnl_sol), 0.0) as total_pnl_sol,
                COALESCE(MAX(pnl_pct), 0.0) as best_trade_pct,
                COALESCE(MIN(pnl_pct), 0.0) as worst_trade_pct,
                COALESCE(AVG(EXTRACT(EPOCH FROM (exited_at - entered_at))/60), 0.0) as avg_hold_minutes,
                COUNT(*) FILTER (WHERE status IN ('confirmed','partial_exit')) as open_positions,
                COALESCE(SUM(jito_tip_lamports), 0) as jito_tips
            FROM trades
            WHERE created_at >= NOW() - INTERVAL '24 hours' AND status != 'cancelled'
        "#).fetch_one(&self.pool).await?;

        let total = row.total_trades.unwrap_or(0) as u32;
        let wins  = row.winning_trades.unwrap_or(0) as u32;
        Ok(SessionStats {
            total_trades: total,
            winning_trades: wins,
            losing_trades: row.losing_trades.unwrap_or(0) as u32,
            win_rate: if total > 0 { wins as f64 / total as f64 } else { 0.0 },
            total_pnl_sol: row.total_pnl_sol.unwrap_or(0.0),
            best_trade_pct: row.best_trade_pct.unwrap_or(0.0),
            worst_trade_pct: row.worst_trade_pct.unwrap_or(0.0),
            avg_hold_minutes: row.avg_hold_minutes.unwrap_or(0.0),
            open_positions: row.open_positions.unwrap_or(0) as u32,
            signals_scanned: 0,
            signals_filtered_out: 0,
            jito_tips_paid_sol: row.jito_tips.unwrap_or(0) as f64 / 1e9,
        })
    }

    pub async fn count_recent_losses(&self, hours: u32) -> Result<i64> {
        let row = sqlx::query!(
            "SELECT COUNT(*) as cnt FROM trades WHERE status='closed' AND pnl_sol < 0
             AND created_at >= NOW() - MAKE_INTERVAL(hours => $1)",
            hours as i32,
        ).fetch_one(&self.pool).await?;
        Ok(row.cnt.unwrap_or(0))
    }

    pub async fn log_signal(&self, signal: &TokenSignal, passed: bool, rejection: Option<&str>) -> Result<()> {
        sqlx::query!(
            r#"INSERT INTO signals (id, mint, source, signal_type, detected_at, filter_passed, filter_rejection)
               VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT DO NOTHING"#,
            signal.id, signal.mint,
            format!("{:?}", signal.source).to_lowercase(),
            format!("{:?}", signal.signal_type).to_lowercase(),
            signal.detected_at, passed, rejection,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn insert_price_candle(&self, mint: &str, candle: &Candle) -> Result<()> {
        sqlx::query!(
            r#"INSERT INTO price_candles_2s (mint, ts, open, high, low, close, volume, buy_count, sell_count)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) ON CONFLICT DO NOTHING"#,
            mint, candle.timestamp, candle.open, candle.high, candle.low,
            candle.close, candle.volume, candle.buy_count as i32, candle.sell_count as i32,
        ).execute(&self.pool).await?;
        Ok(())
    }
}
