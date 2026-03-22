use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::BotConfig;
use crate::db::{Database, RedisClient};
use crate::filter::FilterPipeline;
use crate::types::*;

pub struct StrategyEngine {
    config: Arc<BotConfig>,
    db: Arc<Database>,
    redis: Arc<RedisClient>,
    filter: Arc<FilterPipeline>,
    trade_tx: mpsc::Sender<TradeDecision>,
}

impl StrategyEngine {
    pub fn new(
        config: Arc<BotConfig>, db: Arc<Database>, redis: Arc<RedisClient>,
        filter: Arc<FilterPipeline>, trade_tx: mpsc::Sender<TradeDecision>,
    ) -> Self {
        Self { config, db, redis, filter, trade_tx }
    }

    pub async fn process_signal(&self, signal: TokenSignal) -> Result<()> {
        let _ = self.redis.increment_signals_scanned().await;

        if self.is_circuit_broken().await? {
            debug!("Circuit breaker active — skipping {}", &signal.mint[..8]);
            return Ok(());
        }

        let hour = chrono::Utc::now().hour();
        let w = &self.config.bot;
        if hour < w.trading_window_start_utc || hour >= w.trading_window_end_utc {
            return Ok(());
        }

        let open = self.redis.get_open_position_count().await?;
        if open >= self.config.wallet.max_concurrent_positions {
            debug!("Max positions reached ({})", open);
            return Ok(());
        }

        let at_risk   = self.redis.get_capital_at_risk_sol().await?;
        let available = self.config.wallet.max_total_capital_sol - at_risk - self.config.wallet.reserve_sol;
        if available < 0.05 {
            debug!("Insufficient capital ({:.4} SOL available)", available);
            return Ok(());
        }

        if self.redis.has_active_position(&signal.mint).await? {
            debug!("Already have position in {}", &signal.mint[..8]);
            return Ok(());
        }

        let fr = self.filter.evaluate(&signal).await?;
        if !fr.passed {
            debug!("❌ {} rejected: {}", &signal.mint[..8], fr.rejection_reason.as_deref().unwrap_or("?"));
            return Ok(());
        }

        let decision = match signal.signal_type {
            SignalType::SmartWalletBuy =>
                self.build_copy_decision(signal, fr, available).await?,
            SignalType::NewTokenLaunch | SignalType::LiquidityAdded =>
                self.build_snipe_decision(signal, fr, available).await?,
            SignalType::SentimentSpike | SignalType::CoordinatedMention =>
                self.build_sentiment_decision(signal, fr, available).await?,
        };

        if decision.decision_type == DecisionType::Skip { return Ok(()); }

        info!("📝 Decision: {:?} | {} | {:.4} SOL | ML {:.3}",
            decision.decision_type, &decision.signal.mint[..8],
            decision.buy_amount_sol, decision.filter_result.ensemble_score);

        let _ = self.trade_tx.send(decision).await;
        Ok(())
    }

    async fn build_snipe_decision(&self, signal: TokenSignal, fr: FilterResult, available: f64) -> Result<TradeDecision> {
        let cfg    = &self.config.strategy;
        let size   = self.kelly_size(fr.win_probability, available);
        let entry_1 = size * cfg.snipe_entry_1_pct;
        Ok(TradeDecision {
            id: Uuid::new_v4(), signal, filter_result: fr,
            decision_type: DecisionType::Buy, strategy_track: StrategyTrack::Snipe,
            buy_amount_sol: entry_1,
            max_slippage_bps: self.config.execution.slippage_start_bps,
            entry_delay_seconds: 0,
            take_profit_1: cfg.tp_1_multiplier, take_profit_2: cfg.tp_2_multiplier,
            stop_loss: 1.0 - cfg.hard_stop_loss_pct, time_stop_minutes: cfg.time_stop_minutes,
            decided_at: chrono::Utc::now(),
        })
    }

    async fn build_copy_decision(&self, signal: TokenSignal, fr: FilterResult, available: f64) -> Result<TradeDecision> {
        let cfg = &self.config.strategy;
        let Some(ref copy) = signal.copy_trade else { return Ok(self.skip(signal, fr)); };
        if copy.source_wallet_winrate < cfg.copy_wallet_min_winrate { return Ok(self.skip(signal, fr)); }
        let our_buy = (copy.buy_amount_sol * 0.5).min(cfg.copy_max_buy_sol).min(available * 0.25).max(0.01);
        Ok(TradeDecision {
            id: Uuid::new_v4(), signal, filter_result: fr,
            decision_type: DecisionType::Buy, strategy_track: StrategyTrack::CopyTrade,
            buy_amount_sol: our_buy, max_slippage_bps: self.config.execution.slippage_start_bps,
            entry_delay_seconds: 0, take_profit_1: 1.5, take_profit_2: 2.5,
            stop_loss: 0.80, time_stop_minutes: 45, decided_at: chrono::Utc::now(),
        })
    }

    async fn build_sentiment_decision(&self, signal: TokenSignal, fr: FilterResult, available: f64) -> Result<TradeDecision> {
        let size = (available * 0.05).min(0.15).min(self.config.wallet.max_position_size_sol);
        Ok(TradeDecision {
            id: Uuid::new_v4(), signal, filter_result: fr,
            decision_type: DecisionType::Buy, strategy_track: StrategyTrack::Sentiment,
            buy_amount_sol: size, max_slippage_bps: 800,
            entry_delay_seconds: 10, take_profit_1: 1.3, take_profit_2: 1.8,
            stop_loss: 0.85, time_stop_minutes: 30, decided_at: chrono::Utc::now(),
        })
    }

    pub async fn manage_positions(&self) -> Result<()> {
        let positions = self.redis.get_all_open_positions().await?;
        let cfg = &self.config.strategy;

        for pos in positions {
            let price = self.get_price(&pos.mint).await;
            if price <= 0.0 { continue; }
            let entry = pos.entry_price_usd;
            if entry <= 0.0 { continue; }

            let mult = price / entry;
            let age_min = pos.entered_at
                .map(|t| chrono::Utc::now().signed_duration_since(t).num_minutes() as u32)
                .unwrap_or(0);

            if mult >= cfg.tp_1_multiplier && !self.redis.has_taken_tp1(&pos.mint).await? {
                info!("🎯 TP1 {:.2}x on {} — selling {:.0}%", mult, &pos.mint[..8], cfg.tp_1_sell_pct * 100.0);
                self.fire_partial_sell(&pos, cfg.tp_1_sell_pct).await?;
                self.redis.mark_tp1_taken(&pos.mint).await?;
            }

            if mult >= cfg.tp_2_multiplier && !self.redis.has_taken_tp2(&pos.mint).await? {
                info!("🎯 TP2 {:.2}x on {} — selling {:.0}%, setting trail", mult, &pos.mint[..8], cfg.tp_2_sell_pct * 100.0);
                self.fire_partial_sell(&pos, cfg.tp_2_sell_pct).await?;
                self.redis.mark_tp2_taken(&pos.mint).await?;
                let trail = price * (1.0 - cfg.tp_3_trailing_stop_pct);
                self.redis.set_trailing_stop(&pos.mint, trail).await?;
            }

            if mult <= (1.0 - cfg.hard_stop_loss_pct) {
                warn!("🛑 Stop loss {:.2}x on {} — full exit", mult, &pos.mint[..8]);
                self.fire_full_exit(&pos).await?;
                self.check_circuit_breaker().await?;
                continue;
            }

            if let Some(trail) = self.redis.get_trailing_stop(&pos.mint).await? {
                let new_trail = price * (1.0 - cfg.tp_3_trailing_stop_pct);
                if new_trail > trail { self.redis.set_trailing_stop(&pos.mint, new_trail).await?; }
                if price <= trail {
                    info!("📉 Trail stop hit on {} at {:.2}x", &pos.mint[..8], mult);
                    self.fire_full_exit(&pos).await?;
                    continue;
                }
            }

            if age_min >= cfg.time_stop_minutes && mult < cfg.tp_1_multiplier {
                warn!("⏰ Time stop {}min on {} at {:.2}x", age_min, &pos.mint[..8], mult);
                self.fire_full_exit(&pos).await?;
            }
        }
        Ok(())
    }

    fn kelly_size(&self, win_prob: f64, available: f64) -> f64 {
        let b = 1.0; let q = 1.0 - win_prob;
        let kelly = (win_prob * b - q) / b;
        let half_kelly = (kelly * 0.5).max(0.0).min(0.25);
        (available * half_kelly).min(self.config.wallet.max_position_size_sol).max(0.05)
    }

    async fn fire_partial_sell(&self, pos: &Trade, pct: f64) -> Result<()> {
        let _ = self.trade_tx.send(self.sell_decision(pos, pct)).await;
        Ok(())
    }

    async fn fire_full_exit(&self, pos: &Trade) -> Result<()> {
        self.fire_partial_sell(pos, 1.0).await?;
        self.redis.remove_open_position(&pos.mint).await?;
        Ok(())
    }

    fn sell_decision(&self, pos: &Trade, pct: f64) -> TradeDecision {
        TradeDecision {
            id: Uuid::new_v4(),
            signal: TokenSignal {
                id: Uuid::new_v4(), mint: pos.mint.clone(),
                source: SignalSource::Yellowstone, signal_type: SignalType::NewTokenLaunch,
                detected_at: chrono::Utc::now(), on_chain: None, social: None, copy_trade: None,
            },
            filter_result: pos.filter_result.clone(),
            decision_type: DecisionType::PartialSell { pct },
            strategy_track: pos.strategy_track.clone(),
            buy_amount_sol: 0.0, max_slippage_bps: 1500,
            entry_delay_seconds: 0, take_profit_1: 0.0, take_profit_2: 0.0,
            stop_loss: 0.0, time_stop_minutes: 0, decided_at: chrono::Utc::now(),
        }
    }

    fn skip(&self, signal: TokenSignal, fr: FilterResult) -> TradeDecision {
        TradeDecision {
            id: Uuid::new_v4(), signal, filter_result: fr,
            decision_type: DecisionType::Skip, strategy_track: StrategyTrack::Snipe,
            buy_amount_sol: 0.0, max_slippage_bps: 0,
            entry_delay_seconds: 0, take_profit_1: 0.0, take_profit_2: 0.0,
            stop_loss: 0.0, time_stop_minutes: 0, decided_at: chrono::Utc::now(),
        }
    }

    async fn is_circuit_broken(&self) -> Result<bool> {
        Ok(self.redis.get_circuit_breaker_active().await.unwrap_or(false))
    }

    async fn check_circuit_breaker(&self) -> Result<()> {
        let losses = self.db.count_recent_losses(1).await?;
        let cfg = &self.config.strategy;
        if losses >= cfg.circuit_breaker_consecutive_losses as i64 {
            warn!("⚡ Circuit breaker: {} losses in 1h — pausing {}min", losses, cfg.circuit_breaker_pause_minutes);
            self.redis.set_circuit_breaker_active(cfg.circuit_breaker_pause_minutes * 60).await?;
        }
        Ok(())
    }

    async fn get_price(&self, mint: &str) -> f64 {
        self.redis.get_cached_price(mint).await.unwrap_or(0.0)
    }
}
