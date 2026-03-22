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
        // FIX: cast all DECIMAL/NUMERIC params to float8 so sqlx doesn't
        // infer BigDecimal (which requires the optional "bigdecimal" feature).
        sqlx::query!(
            r#"INSERT INTO trades (
                id, mint, strategy_track, status,
                entry_price_usd, entry_amount_sol,
                ml_tabular_score, ml_transformer_score, ml_gnn_score, ml_nlp_score,
                ml_ensemble_score, jito_tip_lamports, created_at
            ) VALUES ($1,$2,$3,$4,
                $5::float8,$6::float8,
                $7::float8,$8::float8,$9::float8,$10::float8,
                $11::float8,$12,$13)
            ON CONFLICT (id) DO NOTHING"#,
            trade.id,
            trade.mint,
            format!("{:?}", trade.strategy_track).to_lowercase(),
            format!("{:?}", trade.status).to_lowercase(),
            trade.entry_price_usd,
            trade.entry_amount_sol,
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

    pub async fn update_trade_failed(&self, id: Uuid) -> Result<()> {
        sqlx::query!(
            "UPDATE trades SET status = 'failed' WHERE id = $1", id
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn log_dry_run_trade(&self, decision: &TradeDecision) -> Result<()> {
        sqlx::query!(
            r#"INSERT INTO trades
               (id, mint, strategy_track, status, entry_amount_sol, ml_ensemble_score, created_at)
               VALUES ($1,$2,$3,'cancelled',$4::float8,$5::float8,NOW())
               ON CONFLICT DO NOTHING"#,
            decision.id,
            decision.signal.mint,
            format!("{:?}", decision.strategy_track).to_lowercase(),
            decision.buy_amount_sol,
            decision.filter_result.ensemble_score,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn log_filter_result(&self, mint: &str, result: &FilterResult) -> Result<()> {
        sqlx::query!(
            r#"INSERT INTO tokens (mint, filter_passed, filter_rejection_reason, ml_ensemble_score)
               VALUES ($1,$2,$3,$4::float8)
               ON CONFLICT (mint) DO UPDATE SET
                 filter_passed = EXCLUDED.filter_passed,
                 filter_rejection_reason = EXCLUDED.filter_rejection_reason,
                 ml_ensemble_score = EXCLUDED.ml_ensemble_score,
                 updated_at = NOW()"#,
            mint,
            result.passed,
            result.rejection_reason.as_deref(),
            result.ensemble_score,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn get_deployer_rug_count(&self, wallet: &str) -> Result<Option<u32>> {
        let row = sqlx::query!(
            "SELECT rug_count FROM deployer_wallets WHERE wallet = $1", wallet
        ).fetch_optional(&self.pool).await?;
        // FIX: rug_count is INTEGER → Option<i32> from sqlx; can't cast with `as u32` directly
        Ok(row.and_then(|r| r.rug_count).map(|n| n as u32))
    }

    pub async fn get_copy_wallet_winrate(&self, wallet: &str) -> Result<Option<f64>> {
        // FIX: win_rate is DECIMAL → cast to float8 in SQL so sqlx returns f64
        let row = sqlx::query!(
            "SELECT win_rate::float8 AS win_rate FROM copy_wallets WHERE wallet = $1 AND is_active = true",
            wallet
        ).fetch_optional(&self.pool).await?;
        Ok(row.and_then(|r| r.win_rate))
    }

    // ── Session stats ─────────────────────────────────────────────────────────
    // FIX: every DECIMAL/NUMERIC expression cast to ::float8 so sqlx resolves
    // them as f64 rather than requiring the bigdecimal optional feature.
    pub async fn get_session_stats(&self) -> Result<SessionStats> {
        let row = sqlx::query!(r#"
            SELECT
                COUNT(*)::bigint                                                    AS total_trades,
                SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END)::bigint             AS winning_trades,
                SUM(CASE WHEN pnl_sol <= 0 AND status='closed' THEN 1 ELSE 0 END)::bigint
                                                                                   AS losing_trades,
                COALESCE(SUM(pnl_sol)::float8,  0.0)                              AS total_pnl_sol,
                COALESCE(MAX(pnl_pct)::float8,  0.0)                              AS best_trade_pct,
                COALESCE(MIN(pnl_pct)::float8,  0.0)                              AS worst_trade_pct,
                COALESCE(AVG(EXTRACT(EPOCH FROM (exited_at - entered_at)) / 60.0)::float8, 0.0)
                                                                                   AS avg_hold_minutes,
                COUNT(*) FILTER (WHERE status IN ('confirmed','partial_exit'))::bigint
                                                                                   AS open_positions,
                COALESCE(SUM(jito_tip_lamports)::bigint, 0)                       AS jito_tips
            FROM trades
            WHERE created_at >= NOW() - INTERVAL '24 hours'
              AND status != 'cancelled'
        "#).fetch_one(&self.pool).await?;

        let total = row.total_trades.unwrap_or(0) as u32;
        let wins  = row.winning_trades.unwrap_or(0) as u32;
        Ok(SessionStats {
            total_trades:        total,
            winning_trades:      wins,
            losing_trades:       row.losing_trades.unwrap_or(0) as u32,
            win_rate:            if total > 0 { wins as f64 / total as f64 } else { 0.0 },
            total_pnl_sol:       row.total_pnl_sol.unwrap_or(0.0),
            best_trade_pct:      row.best_trade_pct.unwrap_or(0.0),
            worst_trade_pct:     row.worst_trade_pct.unwrap_or(0.0),
            avg_hold_minutes:    row.avg_hold_minutes.unwrap_or(0.0),
            open_positions:      row.open_positions.unwrap_or(0) as u32,
            signals_scanned:     0,
            signals_filtered_out: 0,
            jito_tips_paid_sol:  row.jito_tips.unwrap_or(0) as f64 / 1e9,
        })
    }

    // ── Training stats ────────────────────────────────────────────────────────
    pub async fn get_training_stats(&self) -> Result<TrainingStats> {
        let token_row = sqlx::query!(r#"
            SELECT
                COUNT(*)::bigint                                                        AS total_tokens,
                COUNT(*) FILTER (WHERE ml_label IS NOT NULL)::bigint                   AS labeled_tokens,
                COUNT(*) FILTER (WHERE ml_label = 1)::bigint                           AS positive_labels,
                COUNT(*) FILTER (WHERE ml_label = 0)::bigint                           AS negative_labels,
                COUNT(*) FILTER (WHERE filter_passed = true)::bigint                   AS tokens_passed,
                COALESCE(
                    EXTRACT(EPOCH FROM (MAX(first_seen_at) - MIN(first_seen_at)))::float8 / 3600.0,
                    0.0
                )                                                                      AS hours_of_data
            FROM tokens
        "#).fetch_one(&self.pool).await?;

        let sig_row = sqlx::query!(r#"
            SELECT
                COUNT(*)::bigint                                                AS total_signals,
                COUNT(*) FILTER (WHERE source = 'twitter')::bigint             AS twitter,
                COUNT(*) FILTER (WHERE source = 'telegram')::bigint            AS telegram,
                COUNT(*) FILTER (WHERE source = 'yellowstone')::bigint         AS yellowstone,
                COUNT(*) FILTER (WHERE source = 'copy_trade')::bigint          AS copy_trade
            FROM signals
        "#).fetch_one(&self.pool).await?;

        // FIX: training_sessions may not exist until migration 003 runs.
        // Use query_unchecked! so we don't get a compile-time DB verification error.
        let session_ts: Option<chrono::DateTime<chrono::Utc>> = sqlx::query_scalar_unchecked!(
            "SELECT started_at FROM training_sessions ORDER BY started_at ASC LIMIT 1"
        ).fetch_optional(&self.pool).await.unwrap_or(None);

        let rej_rows = sqlx::query!(r#"
            SELECT
                COALESCE(filter_rejection_reason, 'Unknown') AS reason,
                COUNT(*)::bigint                             AS cnt
            FROM tokens
            WHERE filter_rejection_reason IS NOT NULL
            GROUP BY filter_rejection_reason
            ORDER BY cnt DESC
            LIMIT 8
        "#).fetch_all(&self.pool).await?;

        let total  = token_row.total_tokens.unwrap_or(0);
        let passed = token_row.tokens_passed.unwrap_or(0);
        let hours  = token_row.hours_of_data.unwrap_or(0.0);

        Ok(TrainingStats {
            total_tokens:           total,
            labeled_tokens:         token_row.labeled_tokens.unwrap_or(0),
            positive_labels:        token_row.positive_labels.unwrap_or(0),
            negative_labels:        token_row.negative_labels.unwrap_or(0),
            hours_of_data:          hours,
            collection_started_at:  session_ts,
            total_signals:          sig_row.total_signals.unwrap_or(0),
            twitter_signals:        sig_row.twitter.unwrap_or(0),
            telegram_signals:       sig_row.telegram.unwrap_or(0),
            yellowstone_signals:    sig_row.yellowstone.unwrap_or(0),
            copy_trade_signals:     sig_row.copy_trade.unwrap_or(0),
            tokens_passed_filter:   passed,
            tokens_per_hour:        if hours > 0.0 { total as f64 / hours } else { 0.0 },
            filter_pass_rate:       if total > 0 { passed as f64 / total as f64 } else { 0.0 },
            top_rejections: rej_rows.into_iter().map(|r| RejectionStat {
                reason: r.reason.unwrap_or_default(),
                count:  r.cnt.unwrap_or(0),
            }).collect(),
        })
    }

    pub async fn count_consecutive_losses(&self) -> Result<i64> {
        let rows = sqlx::query!(
            r#"SELECT pnl_sol::float8 AS pnl_sol
               FROM trades
               WHERE status = 'closed' AND exited_at IS NOT NULL
               ORDER BY exited_at DESC
               LIMIT 10"#
        ).fetch_all(&self.pool).await?;

        let mut count = 0i64;
        for row in rows {
            match row.pnl_sol {
                Some(pnl) if pnl < 0.0 => count += 1,
                _ => break,
            }
        }
        Ok(count)
    }

    pub async fn count_recent_losses(&self, hours: u32) -> Result<i64> {
        let row = sqlx::query!(
            r#"SELECT COUNT(*)::bigint AS cnt FROM trades
               WHERE status='closed' AND pnl_sol < 0
               AND created_at >= NOW() - MAKE_INTERVAL(hours => $1)"#,
            hours as i32,
        ).fetch_one(&self.pool).await?;
        Ok(row.cnt.unwrap_or(0))
    }

    pub async fn log_signal(&self, signal: &TokenSignal, passed: bool, rejection: Option<&str>) -> Result<()> {
        let source_wallet: Option<&str> = signal.copy_trade.as_ref().map(|ct| ct.source_wallet.as_str());
        sqlx::query!(
            r#"INSERT INTO signals
                   (id, mint, source, signal_type, detected_at,
                    filter_passed, filter_rejection, source_wallet)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
               ON CONFLICT DO NOTHING"#,
            signal.id,
            signal.mint,
            format!("{:?}", signal.source).to_lowercase(),
            format!("{:?}", signal.signal_type).to_lowercase(),
            signal.detected_at,
            passed,
            rejection,
            source_wallet,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn insert_price_candle(&self, mint: &str, candle: &Candle) -> Result<()> {
        // FIX: cast all DECIMAL params to float8
        sqlx::query!(
            r#"INSERT INTO price_candles_2s
               (mint, ts, open, high, low, close, volume, buy_count, sell_count)
               VALUES ($1,$2,$3::float8,$4::float8,$5::float8,$6::float8,$7::float8,$8,$9)
               ON CONFLICT DO NOTHING"#,
            mint,
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.buy_count as i32,
            candle.sell_count as i32,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn start_training_session(&self) -> Result<i32> {
        // FIX: use query_unchecked! — training_sessions may not exist at compile
        // time if migration 003 hasn't been verified by sqlx prepare yet.
        let row: (i32,) = sqlx::query_as_unchecked!(
            (i32,),
            "INSERT INTO training_sessions (started_at) VALUES (NOW()) RETURNING id"
        ).fetch_one(&self.pool).await?;
        Ok(row.0)
    }
}
