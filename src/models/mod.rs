//! models/mod.rs — ONNX ensemble inference.
//!
//! Each model is loaded from disk at startup.  If a model file is absent
//! (before the first training run) the corresponding scorer falls back to a
//! cheap deterministic heuristic so the bot can still operate in dry mode.
//!
//! Feature order in `extract_tabular_features` is byte-for-byte identical to
//! `TABULAR_FEATURES` in `ml/scripts/train_all.py` — keep them in sync.

use anyhow::Result;
use chrono::Timelike;
use ndarray::{Array2, Array3};
use ort::{GraphOptimizationLevel, Session};
use std::f32::consts::PI as PI32;
use std::sync::Arc;
use tracing::{info, warn};

use crate::config::ModelsConfig;
use crate::types::*;

// ── Constants (must match train_all.py / train_gnn.py / train_nlp.py) ─────────
const N_TABULAR:   usize = 37;
const SEQ_LEN:     usize = 30;
const SEQ_FEAT:    usize = 6;
const NODE_DIM:    usize = 12;
const MAX_NLP_LEN: usize = 128;

// ── Model ensemble ────────────────────────────────────────────────────────────

pub struct ModelEnsemble {
    config:    ModelsConfig,
    tabular:   Option<Arc<Session>>,
    transformer: Option<Arc<Session>>,
    gnn:       Option<Arc<Session>>,
    nlp:       Option<Arc<Session>>,
    tokenizer: Option<Arc<tokenizers::Tokenizer>>,
}

impl ModelEnsemble {
    pub fn load(config: &ModelsConfig) -> Result<Self> {
        let tabular     = load_session(&config.tabular_model_path,     "tabular");
        let transformer = load_session(&config.transformer_model_path, "transformer");
        let gnn         = load_session(&config.gnn_model_path,         "gnn");
        let nlp_sess    = load_session(&config.nlp_model_path,         "nlp");

        // The HuggingFace tokenizer is saved beside the NLP ONNX model.
        let tok_path = {
            let parent = std::path::Path::new(&config.nlp_model_path)
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."));
            format!("{}/finbert_tokenizer/tokenizer.json", parent.display())
        };
        let tokenizer = if std::path::Path::new(&tok_path).exists() {
            match tokenizers::Tokenizer::from_file(&tok_path) {
                Ok(t)  => { info!("✅ NLP tokenizer loaded from {}", tok_path); Some(Arc::new(t)) }
                Err(e) => { warn!("Tokenizer load failed ({}): {} — heuristic active", tok_path, e); None }
            }
        } else {
            warn!("NLP tokenizer not found at {} — NLP will use heuristic until train_nlp.py is run", tok_path);
            None
        };

        Ok(Self {
            config: config.clone(),
            tabular,
            transformer,
            gnn,
            nlp: nlp_sess,
            tokenizer,
        })
    }

    // ── Tabular (CatBoost ONNX) ───────────────────────────────────────────────

    pub async fn score_tabular(&self, signal: &TokenSignal) -> Result<f64> {
        let Some(ref session) = self.tabular else {
            return self.tabular_heuristic(signal);
        };
        let sess = Arc::clone(session);
        let feats = extract_tabular_features(signal);
        tokio::task::spawn_blocking(move || {
            let arr = Array2::<f32>::from_shape_vec((1, N_TABULAR), feats)
                .map_err(|e| anyhow::anyhow!("Array shape error: {}", e))?;
            let outputs = sess.run(ort::inputs![arr.view()]?)
                .map_err(|e| anyhow::anyhow!("Tabular ONNX run failed: {}", e))?;
            // CatBoost native ONNX: outputs[0] = label int64, outputs[1] = probs [N,2]
            // Fall back to outputs[0] if only one output exists.
            let prob = if outputs.len() >= 2 {
                let t = outputs[1].try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("Extract probs failed: {}", e))?;
                let v = t.view();
                if v.ndim() == 2 && v.shape()[1] >= 2 { v[[0, 1]] as f64 }
                else { v.iter().copied().next().unwrap_or(0.5) as f64 }
            } else if !outputs.is_empty() {
                let t = outputs[0].try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("Extract label failed: {}", e))?;
                t.view().iter().copied().next().unwrap_or(0.5) as f64
            } else {
                return Err(anyhow::anyhow!("Tabular model produced no outputs"));
            };
            Ok(prob.clamp(0.0, 1.0))
        }).await?
    }

    // ── Price Transformer ONNX ────────────────────────────────────────────────

    pub async fn score_transformer(&self, signal: &TokenSignal) -> Result<f64> {
        let candles = signal.on_chain.as_ref()
            .map(|d| d.price_candles_2s.as_slice())
            .unwrap_or(&[]);

        let Some(ref session) = self.transformer else {
            return self.transformer_heuristic(candles);
        };
        if candles.len() < 5 {
            return self.transformer_heuristic(candles);
        }

        let candles_owned = candles.to_vec();
        let sess = Arc::clone(session);
        tokio::task::spawn_blocking(move || {
            let seq = build_price_sequence(&candles_owned)
                .ok_or_else(|| anyhow::anyhow!("Could not build price sequence"))?;
            let outputs = sess.run(ort::inputs![seq.view()]?)
                .map_err(|e| anyhow::anyhow!("Transformer ONNX run failed: {}", e))?;
            let t = outputs[0].try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Transformer extract failed: {}", e))?;
            let v = t.view();
            // Output is [1, 1] sigmoid logit
            let score = v.iter().copied().next().unwrap_or(0.5) as f64;
            Ok(score.clamp(0.0, 1.0))
        }).await?
    }

    // ── Wallet GNN ONNX ───────────────────────────────────────────────────────

    pub async fn score_gnn(&self, signal: &TokenSignal) -> Result<f64> {
        let Some(ref session) = self.gnn else {
            return self.gnn_heuristic(signal);
        };
        let Some(ref on_chain) = signal.on_chain else {
            return Ok(0.5);
        };
        let node_feats = build_deployer_node_features(on_chain);
        let sess = Arc::clone(session);
        tokio::task::spawn_blocking(move || {
            // Wrapper exported by train_gnn.py: input "node_features" [N, 12]
            // output "rug_probability" [N]; we pass a single deployer-wallet row.
            let arr = Array2::<f32>::from_shape_vec((1, NODE_DIM), node_feats.to_vec())
                .map_err(|e| anyhow::anyhow!("GNN array shape error: {}", e))?;
            let outputs = sess.run(ort::inputs!["node_features" => arr.view()]?)
                .map_err(|e| anyhow::anyhow!("GNN ONNX run failed: {}", e))?;
            let t = outputs[0].try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("GNN extract failed: {}", e))?;
            let rug_prob = t.view().iter().copied().next().unwrap_or(0.5) as f64;
            // Invert: high rug_probability → low trade score
            Ok((1.0 - rug_prob).clamp(0.0, 1.0))
        }).await?
    }

    // ── FinBERT NLP ONNX ──────────────────────────────────────────────────────

    pub async fn score_nlp(&self, signal: &TokenSignal) -> Result<f64> {
        // Need both the ONNX session and the tokenizer.
        let (Some(ref session), Some(ref tokenizer)) = (&self.nlp, &self.tokenizer) else {
            return self.nlp_heuristic(signal);
        };
        let text = signal.social.as_ref()
            .and_then(|s| s.message_samples.first().cloned())
            .unwrap_or_default();
        if text.is_empty() {
            return self.nlp_heuristic(signal);
        }

        let sess = Arc::clone(session);
        let tok  = Arc::clone(tokenizer);
        tokio::task::spawn_blocking(move || {
            let enc = tok.encode(text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
            let ids   = pad_ids(enc.get_ids());
            let mask  = pad_ids(enc.get_attention_mask());
            let types = pad_ids(enc.get_type_ids());
            let ids_arr   = Array2::<i64>::from_shape_vec((1, MAX_NLP_LEN), ids)?;
            let mask_arr  = Array2::<i64>::from_shape_vec((1, MAX_NLP_LEN), mask)?;
            let types_arr = Array2::<i64>::from_shape_vec((1, MAX_NLP_LEN), types)?;
            let outputs = sess.run(ort::inputs![
                "input_ids"      => ids_arr.view(),
                "attention_mask" => mask_arr.view(),
                "token_type_ids" => types_arr.view()
            ]?).map_err(|e| anyhow::anyhow!("NLP ONNX run failed: {}", e))?;
            // SentimentWrapper exports: outputs[0] = sentiment_score [B], outputs[1] = velocity [B]
            let t = outputs[0].try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("NLP extract failed: {}", e))?;
            let score = t.view().iter().copied().next().unwrap_or(0.5) as f64;
            Ok(score.clamp(0.0, 1.0))
        }).await?
    }
}

// ── Heuristics — run when models haven't been trained yet ─────────────────────

impl ModelEnsemble {
    fn tabular_heuristic(&self, signal: &TokenSignal) -> Result<f64> {
        let score = signal.on_chain.as_ref().map(|d| {
            let liq_ok    = (d.liquidity_usd > 50_000.0 && d.liquidity_usd < 2_000_000.0) as u8 as f64 * 0.15;
            let mint_ok   = d.mint_authority_disabled as u8 as f64 * 0.15;
            let freeze_ok = d.freeze_authority_disabled as u8 as f64 * 0.10;
            let lp_ok     = (d.lp_locked && d.lp_lock_days.unwrap_or(0) >= 30) as u8 as f64 * 0.15;
            let dev_ok    = (d.dev_holding_pct < 0.05) as u8 as f64 * 0.10;
            let snip_ok   = (d.sniper_concentration_pct < 0.03) as u8 as f64 * 0.10;
            let rug_ok    = (d.deployer_previous_tokens.iter().all(|t| !t.outcome.is_negative())) as u8 as f64 * 0.15;
            let buy_press = d.buy_count_1h as f64
                / (d.buy_count_1h + d.sell_count_1h + 1) as f64 * 0.10;
            (0.2 + liq_ok + mint_ok + freeze_ok + lp_ok + dev_ok + snip_ok + rug_ok + buy_press).clamp(0.0, 1.0)
        }).unwrap_or(0.5);
        Ok(score)
    }

    fn transformer_heuristic(&self, candles: &[Candle]) -> Result<f64> {
        if candles.len() < 5 { return Ok(0.5); }
        let recent = &candles[candles.len().saturating_sub(5)..];
        let dom = recent.iter().map(|c| {
            let total = (c.buy_count + c.sell_count) as f64;
            if total > 0.0 { c.buy_count as f64 / total } else { 0.5 }
        }).sum::<f64>() / recent.len() as f64;
        Ok(dom.clamp(0.0, 1.0))
    }

    fn gnn_heuristic(&self, signal: &TokenSignal) -> Result<f64> {
        let score = signal.on_chain.as_ref().map(|d| {
            let age_score   = (d.deployer_wallet_age_days as f64 / 365.0).min(1.0) * 0.4;
            let snip_pen    = d.sniper_concentration_pct * 2.0;
            let rug_pen     = d.deployer_previous_tokens.iter()
                .filter(|t| t.outcome.is_negative()).count() as f64 * 0.2;
            (0.6 + age_score - snip_pen - rug_pen).clamp(0.0, 1.0)
        }).unwrap_or(0.5);
        Ok(score)
    }

    fn nlp_heuristic(&self, signal: &TokenSignal) -> Result<f64> {
        Ok(signal.social.as_ref().map(|s| {
            let base  = s.sentiment_score;
            let accel = s.sentiment_acceleration.clamp(-0.5, 0.5) * 0.2;
            let kol   = if s.kol_mention { 0.1 } else { 0.0 };
            (base + accel + kol).clamp(0.0, 1.0)
        }).unwrap_or(0.5))
    }
}

// ── Feature extraction ────────────────────────────────────────────────────────

/// Extracts the 37 tabular features in exactly the same order as
/// `TABULAR_FEATURES` in `ml/scripts/train_all.py`.
fn extract_tabular_features(signal: &TokenSignal) -> Vec<f32> {
    let now = chrono::Utc::now();
    let hour = now.hour() as f32;
    let hour_sin = (2.0 * PI32 * hour / 24.0).sin();
    let hour_cos = (2.0 * PI32 * hour / 24.0).cos();
    let day_of_week = now.weekday().num_days_from_monday() as f32;

    // Unpack on-chain fields
    let (
        liquidity_usd, market_cap_usd, token_age_secs,
        buy_1h, sell_1h, vol_1h, pc5m, pc1h,
        dev_hold, top10, sniper,
        mint_dis, freeze_dis, lp_locked_f, lp_days,
        dep_age, dep_prev, dep_rug_rate,
        is_pf, is_ry, is_mt,
    ) = if let Some(ref d) = signal.on_chain {
        let rug = d.deployer_previous_tokens.iter().filter(|t| t.outcome.is_negative()).count() as f32;
        let tot = d.deployer_previous_tokens.len() as f32;
        (
            d.liquidity_usd as f32,
            d.market_cap_usd as f32,
            d.token_age_seconds as f32,
            d.buy_count_1h as f32,
            d.sell_count_1h as f32,
            d.volume_usd_1h as f32,
            d.price_change_5m_pct as f32,
            d.price_change_1h_pct as f32,
            d.dev_holding_pct as f32,
            d.top_10_holder_pct as f32,
            d.sniper_concentration_pct as f32,
            d.mint_authority_disabled as u8 as f32,
            d.freeze_authority_disabled as u8 as f32,
            d.lp_locked as u8 as f32,
            d.lp_lock_days.unwrap_or(0) as f32,
            d.deployer_wallet_age_days as f32,
            tot,
            if tot > 0.0 { rug / tot } else { 0.0 },
            matches!(d.dex, DexType::PumpFun | DexType::PumpSwap) as u8 as f32,
            matches!(d.dex, DexType::RaydiumAMM | DexType::RaydiumCPMM | DexType::RaydiumCLMM) as u8 as f32,
            matches!(d.dex, DexType::Meteora) as u8 as f32,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };

    let (tw5m, tg5m, sent, sent_acc, kol) = if let Some(ref s) = signal.social {
        (
            s.twitter_mentions_5m  as f32,
            s.telegram_mentions_5m as f32,
            s.sentiment_score      as f32,
            s.sentiment_acceleration as f32,
            s.kol_mention as u8    as f32,
        )
    } else { (0.0, 0.0, 0.5, 0.0, 0.0) };

    // Engineered features (identical to engineer() in train_all.py)
    let buy_sell_ratio = buy_1h / (buy_1h + sell_1h + 1.0);
    let liq_to_mcap   = if market_cap_usd > 0.0 { liquidity_usd / market_cap_usd } else { 0.0 };
    let vol_to_liq    = if liquidity_usd > 0.0 { vol_1h / liquidity_usd } else { 0.0 };
    let gini          = top10.powf(1.5);
    let liq_log       = (1.0 + liquidity_usd).ln();
    let mcap_log      = (1.0 + market_cap_usd).ln();
    let vol_log       = (1.0 + vol_1h).ln();
    let smart_launch  = if mint_dis > 0.5 && freeze_dis > 0.5 && lp_locked_f > 0.5 { 1.0_f32 } else { 0.0 };

    vec![
        // 0-2
        liquidity_usd, market_cap_usd, token_age_secs,
        // 3-6
        buy_1h, sell_1h, buy_sell_ratio, vol_1h,
        // 7-8
        pc5m, pc1h,
        // 9-11
        dev_hold, top10, sniper,
        // 12-15
        mint_dis, freeze_dis, lp_locked_f, lp_days,
        // 16-18
        dep_age, dep_prev, dep_rug_rate,
        // 19-21
        liq_to_mcap, vol_to_liq, gini,
        // 22-26
        tw5m, tg5m, sent, sent_acc, kol,
        // 27-29
        is_pf, is_ry, is_mt,
        // 30-32
        hour_sin, hour_cos, day_of_week,
        // 33-36
        liq_log, mcap_log, vol_log, smart_launch,
    ]
}

/// Builds a `[1, SEQ_LEN, SEQ_FEAT]` tensor from the most recent price candles.
/// Features per step: open, high, low, close, volume, buy_ratio.
fn build_price_sequence(candles: &[Candle]) -> Option<Array3<f32>> {
    if candles.is_empty() { return None; }
    let start = candles.len().saturating_sub(SEQ_LEN);
    let slice = &candles[start..];
    let pad   = SEQ_LEN.saturating_sub(slice.len());
    let mut data = vec![0.0f32; SEQ_LEN * SEQ_FEAT];
    for (i, c) in slice.iter().enumerate() {
        let tot  = (c.buy_count + c.sell_count) as f32;
        let buyr = if tot > 0.0 { c.buy_count as f32 / tot } else { 0.5 };
        let base = (pad + i) * SEQ_FEAT;
        data[base]     = c.open   as f32;
        data[base + 1] = c.high   as f32;
        data[base + 2] = c.low    as f32;
        data[base + 3] = c.close  as f32;
        data[base + 4] = c.volume as f32;
        data[base + 5] = buyr;
    }
    Array3::from_shape_vec((1, SEQ_LEN, SEQ_FEAT), data).ok()
}

/// Builds the 12-element node feature vector for the deployer wallet,
/// matching `node_features()` in `ml/scripts/train_gnn.py`.
fn build_deployer_node_features(d: &OnChainData) -> [f32; NODE_DIM] {
    let rug = d.deployer_previous_tokens.iter().filter(|t| t.outcome.is_negative()).count() as f32;
    let tot = d.deployer_previous_tokens.len() as f32;
    let rug_rate = if tot > 0.0 { rug / tot } else { 0.0 };
    [
        (1.0 + d.deployer_wallet_age_days as f32).ln(), // log1p(wallet_age_days)
        (1.0 + tot * 10.0).ln(),                        // log1p(total_transactions) — proxy
        0.0_f32,                                         // log1p(total_sol_volume)  — unknown
        (rug.min(10.0) / 10.0),                          // rug_involvement / 10
        1.0_f32,                                         // is_deployer = true
        0.0_f32,                                         // is_known_sniper
        0.0_f32,                                         // log1p(sol_balance_at_launch) — unknown
        (1.0 + tot).ln(),                               // log1p(num_tokens_held)
        0.0_f32,                                         // log1p(avg_hold_duration_hours) — unknown
        1.0 - rug_rate,                                  // win_rate proxy
        0.0_f32,                                         // funded_by_exchange — unknown
        0.0_f32,                                         // log1p(hours_since_last_activity) — unknown
    ]
}

/// Pad/truncate a token-id slice to MAX_NLP_LEN and convert to i64.
fn pad_ids<T: Into<i64> + Copy>(ids: &[T]) -> Vec<i64> {
    let mut v: Vec<i64> = ids.iter().take(MAX_NLP_LEN).map(|&x| x.into()).collect();
    v.resize(MAX_NLP_LEN, 0i64);
    v
}

// ── Session loader ────────────────────────────────────────────────────────────

fn load_session(path: &str, name: &str) -> Option<Arc<Session>> {
    if !std::path::Path::new(path).exists() {
        warn!(
            "ONNX model '{}' not found at {} — heuristic scoring active (run training scripts first)",
            name, path,
        );
        return None;
    }
    match Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.commit_from_file(path))
    {
        Ok(s)  => { info!("✅ ONNX model loaded: {} ({})", name, path); Some(Arc::new(s)) }
        Err(e) => {
            warn!("Failed to load ONNX model '{}' at {}: {} — heuristic active", name, path, e);
            None
        }
    }
}

// ── Scheduled retraining ──────────────────────────────────────────────────────

pub async fn retrain_models() -> Result<()> {
    let output = tokio::process::Command::new(".venv/bin/python")
        .arg("ml/scripts/train_all.py")
        .arg("--db-url").arg(std::env::var("DATABASE_URL").unwrap_or_default())
        .arg("--output-dir").arg("./ml/models")
        .output().await?;
    if !output.status.success() {
        anyhow::bail!(
            "Retraining failed:\n{}",
            String::from_utf8_lossy(&output.stderr),
        );
    }
    tracing::info!("✅ Model retraining complete");
    Ok(())
}
