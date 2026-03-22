use anyhow::Result;
use tracing::{info, warn};

use crate::config::ModelsConfig;
use crate::types::*;

/// ModelEnsemble — loads ONNX models at startup.
/// If a model file is missing, returns 0.5 (neutral) so the bot can
/// still operate during the initial dry-run data-collection phase.
pub struct ModelEnsemble {
    config: ModelsConfig,
    // ort::Session handles added here once ONNX files are trained and present
}

impl ModelEnsemble {
    pub fn load(config: &ModelsConfig) -> Result<Self> {
        for (path, name) in [
            (&config.tabular_model_path,     "tabular"),
            (&config.transformer_model_path, "transformer"),
            (&config.gnn_model_path,         "gnn"),
            (&config.nlp_model_path,         "nlp"),
        ] {
            if std::path::Path::new(path).exists() {
                info!("✅ Found model: {} ({})", name, path);
            } else {
                warn!(
                    "Model '{}' not found at {} — returning 0.5 neutral score",
                    name, path,
                );
            }
        }
        Ok(Self { config: config.clone() })
    }

    /// Returns 0.5 until ONNX model is trained and loaded.
    /// Run `python ml/scripts/train_all.py` to generate the model file.
    pub async fn score_tabular(&self, _signal: &TokenSignal) -> Result<f64> {
        Ok(0.5)
    }

    /// Heuristic based on buy/sell candle dominance.
    /// Replaced by the real ONNX transformer after `train_all.py` runs.
    pub async fn score_transformer(&self, signal: &TokenSignal) -> Result<f64> {
        let candles = signal
            .on_chain.as_ref()
            .map(|d| d.price_candles_2s.as_slice())
            .unwrap_or(&[]);
        if candles.len() < 5 { return Ok(0.5); }

        let recent  = &candles[candles.len().saturating_sub(5)..];
        let buy_dom = recent.iter().map(|c| {
            let total = (c.buy_count + c.sell_count) as f64;
            if total > 0.0 { c.buy_count as f64 / total } else { 0.5 }
        }).sum::<f64>() / recent.len() as f64;

        Ok(buy_dom.clamp(0.0, 1.0))
    }

    /// Heuristic based on deployer age and sniper concentration.
    /// Replaced by the real ONNX GNN after `train_gnn.py` runs.
    pub async fn score_gnn(&self, signal: &TokenSignal) -> Result<f64> {
        let score = signal.on_chain.as_ref().map(|d| {
            let age_score      = (d.deployer_wallet_age_days as f64 / 365.0).min(1.0) * 0.4;
            let sniper_penalty = d.sniper_concentration_pct * 2.0;
            let rug_penalty    = d.deployer_previous_tokens.iter()
                .filter(|t| t.outcome == TokenOutcome::Rug)
                .count() as f64 * 0.2;
            (0.6 + age_score - sniper_penalty - rug_penalty).clamp(0.0, 1.0)
        }).unwrap_or(0.5);
        Ok(score)
    }

    /// Uses pre-computed sentiment from the signal scanner.
    /// Replaced by the real ONNX FinBERT model after `train_nlp.py` runs.
    pub async fn score_nlp(&self, signal: &TokenSignal) -> Result<f64> {
        Ok(signal.social.as_ref().map(|s| {
            let base  = s.sentiment_score;
            let accel = s.sentiment_acceleration.clamp(-0.5, 0.5) * 0.2;
            let kol   = if s.kol_mention { 0.1 } else { 0.0 };
            (base + accel + kol).clamp(0.0, 1.0)
        }).unwrap_or(0.5))
    }
}

pub async fn retrain_models() -> Result<()> {
    let output = tokio::process::Command::new(".venv/bin/python")
        .arg("ml/scripts/train_all.py")
        .arg("--db-url")
        .arg(std::env::var("DATABASE_URL").unwrap_or_default())
        .arg("--output-dir")
        .arg("./ml/models")
        .output()
        .await?;

    if !output.status.success() {
        anyhow::bail!(
            "Retraining failed: {}",
            String::from_utf8_lossy(&output.stderr),
        );
    }
    tracing::info!("✅ Model retraining complete");
    Ok(())
}
