use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info};

use crate::config::BotConfig;
use crate::db::{Database, RedisClient};
use crate::models::ModelEnsemble;
use crate::types::*;

pub struct FilterPipeline {
    config: Arc<BotConfig>,
    db: Arc<Database>,
    models: Arc<ModelEnsemble>,
    redis: Arc<RedisClient>,
}

impl FilterPipeline {
    pub fn new(
        config: Arc<BotConfig>,
        db: Arc<Database>,
        models: Arc<ModelEnsemble>,
        redis: Arc<RedisClient>,
    ) -> Self {
        Self { config, db, models, redis }
    }

    pub async fn evaluate(&self, signal: &TokenSignal) -> Result<FilterResult> {
        let cfg = &self.config.filter;

        let mut result = FilterResult {
            passed: false,
            rejection_reason: None,
            liquidity_ok: false, market_cap_ok: false, token_age_ok: false,
            dev_holding_ok: false, sniper_ok: false, mint_authority_ok: false,
            freeze_authority_ok: false, lp_lock_ok: false, creator_history_ok: false,
            price_impact_ok: false,
            tabular_score: 0.0, transformer_score: 0.0, gnn_score: 0.0,
            nlp_score: 0.0, ensemble_score: 0.0, win_probability: 0.0,
        };

        let cache_key = format!("filter:{}", signal.mint);
        if let Some(cached) = self.redis.get_filter_result(&cache_key).await? {
            debug!("Filter cache hit: {}", &signal.mint[..8.min(signal.mint.len())]);
            return Ok(cached);
        }

        let Some(ref on_chain) = signal.on_chain else {
            result.rejection_reason = Some("No on-chain data".into());
            return Ok(result);
        };

        result.liquidity_ok = on_chain.liquidity_usd >= cfg.min_liquidity_usd
                           && on_chain.liquidity_usd <= cfg.max_liquidity_usd;
        if !result.liquidity_ok {
            result.rejection_reason = Some(format!("Liquidity ${:.0} outside range", on_chain.liquidity_usd));
            return Ok(self.cache(result, &cache_key).await);
        }

        result.market_cap_ok = on_chain.market_cap_usd >= cfg.min_market_cap_usd
                            && on_chain.market_cap_usd <= cfg.max_market_cap_usd;
        if !result.market_cap_ok {
            result.rejection_reason = Some(format!("Market cap ${:.0} outside range", on_chain.market_cap_usd));
            return Ok(self.cache(result, &cache_key).await);
        }

        let age_hours = on_chain.token_age_seconds as f64 / 3600.0;
        result.token_age_ok = age_hours <= cfg.max_token_age_hours as f64;
        if !result.token_age_ok {
            result.rejection_reason = Some(format!("Token age {:.1}h exceeds max {}h", age_hours, cfg.max_token_age_hours));
            return Ok(self.cache(result, &cache_key).await);
        }

        result.dev_holding_ok = on_chain.dev_holding_pct <= cfg.max_dev_holding_pct;
        if !result.dev_holding_ok {
            result.rejection_reason = Some(format!("Dev holds {:.1}% > max {:.1}%", on_chain.dev_holding_pct * 100.0, cfg.max_dev_holding_pct * 100.0));
            return Ok(self.cache(result, &cache_key).await);
        }

        result.sniper_ok = on_chain.sniper_concentration_pct <= cfg.max_sniper_concentration_pct;
        if !result.sniper_ok {
            result.rejection_reason = Some(format!("Sniper concentration {:.1}% exceeds {:.1}%", on_chain.sniper_concentration_pct * 100.0, cfg.max_sniper_concentration_pct * 100.0));
            return Ok(self.cache(result, &cache_key).await);
        }

        result.mint_authority_ok = !cfg.require_mint_authority_disabled || on_chain.mint_authority_disabled;
        if !result.mint_authority_ok {
            result.rejection_reason = Some("Mint authority still enabled".into());
            return Ok(self.cache(result, &cache_key).await);
        }

        result.freeze_authority_ok = !cfg.require_freeze_authority_disabled || on_chain.freeze_authority_disabled;
        if !result.freeze_authority_ok {
            result.rejection_reason = Some("Freeze authority still enabled".into());
            return Ok(self.cache(result, &cache_key).await);
        }

        result.lp_lock_ok = on_chain.lp_locked && on_chain.lp_lock_days.unwrap_or(0) >= cfg.min_lp_lock_days;
        if !result.lp_lock_ok {
            result.rejection_reason = Some(format!("LP not locked or lock < {} days", cfg.min_lp_lock_days));
            return Ok(self.cache(result, &cache_key).await);
        }

        let rug_count = self.db.get_deployer_rug_count(&on_chain.deployer_wallet).await?.unwrap_or(0);
        result.creator_history_ok = rug_count <= cfg.max_creator_rug_history;
        if !result.creator_history_ok {
            result.rejection_reason = Some(format!("Deployer has {} previous rugs (max {})", rug_count, cfg.max_creator_rug_history));
            return Ok(self.cache(result, &cache_key).await);
        }

        let sell_amount_usd = on_chain.market_cap_usd * 0.05;
        let reserve         = on_chain.liquidity_usd / 2.0;
        let price_impact    = sell_amount_usd / (reserve + sell_amount_usd);
        result.price_impact_ok = price_impact <= cfg.max_price_impact_5pct_sell;
        if !result.price_impact_ok {
            result.rejection_reason = Some(format!("5% sell would move price {:.1}% (illiquid trap)", price_impact * 100.0));
            return Ok(self.cache(result, &cache_key).await);
        }

        let (tab, trans, gnn, nlp) = tokio::join!(
            self.models.score_tabular(signal),
            self.models.score_transformer(signal),
            self.models.score_gnn(signal),
            self.models.score_nlp(signal),
        );

        result.tabular_score     = tab.unwrap_or(0.5);
        result.transformer_score = trans.unwrap_or(0.5);
        result.gnn_score         = gnn.unwrap_or(0.5);
        result.nlp_score         = nlp.unwrap_or(0.5);

        let m = &self.config.models;
        result.ensemble_score =
            result.tabular_score     * m.tabular_weight
          + result.transformer_score * m.transformer_weight
          + result.gnn_score         * m.gnn_weight
          + result.nlp_score         * m.nlp_weight;
        result.win_probability = result.ensemble_score;

        if result.ensemble_score < cfg.min_win_probability {
            result.rejection_reason = Some(format!(
                "ML score {:.3} < threshold {:.3} | tab={:.2} trans={:.2} gnn={:.2} nlp={:.2}",
                result.ensemble_score, cfg.min_win_probability,
                result.tabular_score, result.transformer_score, result.gnn_score, result.nlp_score,
            ));
            let _ = self.redis.increment_signals_filtered_out().await;
            return Ok(self.cache(result, &cache_key).await);
        }

        result.passed = true;
        info!(
            "✅ PASS {} | ML={:.3} | liq=${:.0} | age={:.1}h | sniper={:.1}%",
            &signal.mint[..8.min(signal.mint.len())],
            result.ensemble_score, on_chain.liquidity_usd, age_hours,
            on_chain.sniper_concentration_pct * 100.0,
        );

        {
            let db   = self.db.clone();
            let mint = signal.mint.clone();
            let r    = result.clone();
            tokio::spawn(async move { let _ = db.log_filter_result(&mint, &r).await; });
        }

        Ok(self.cache(result, &cache_key).await)
    }

    async fn cache(&self, result: FilterResult, key: &str) -> FilterResult {
        let _ = self.redis.set_filter_result(key, &result, 300).await;
        result
    }
}
