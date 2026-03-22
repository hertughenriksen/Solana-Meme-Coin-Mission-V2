pub mod swap_builder;
pub mod rpc_client;

use anyhow::{Context, Result};
use base64::Engine;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::config::BotConfig;
use crate::db::{Database, RedisClient};
use crate::monitor::record_trade_entered;
use crate::types::*;
use rpc_client::SolanaRpcClient;
use swap_builder::*;

pub struct ExecutionEngine {
    config:  Arc<BotConfig>,
    db:      Arc<Database>,
    redis:   Arc<RedisClient>,
    rpc:     Arc<SolanaRpcClient>,
    keypair: solana_sdk::signature::Keypair,
    http:    reqwest::Client,
}

impl ExecutionEngine {
    pub async fn new(config: Arc<BotConfig>, db: Arc<Database>, redis: Arc<RedisClient>) -> Result<Self> {
        let keypair_bytes = std::fs::read(&config.wallet.keypair_path)
            .with_context(|| format!("Cannot read keypair at {}", config.wallet.keypair_path))?;
        let keypair_vec: Vec<u8> = serde_json::from_slice(&keypair_bytes)
            .context("Keypair file must be a JSON array of bytes")?;
        let keypair = solana_sdk::signature::Keypair::from_bytes(&keypair_vec)
            .context("Invalid keypair bytes")?;

        let rpc = Arc::new(SolanaRpcClient::new(config.clone()));
        rpc.spawn_blockhash_refresher();

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(8))
            .tcp_nodelay(true)
            .pool_max_idle_per_host(10)
            .build()?;

        let pubkey = keypair.pubkey().to_string();
        info!("⚡ Execution engine ready | wallet: {}…{}", &pubkey[..4], &pubkey[pubkey.len() - 4..]);
        Ok(Self { config, db, redis, rpc, keypair, http })
    }

    pub async fn run(&self, mut trade_rx: mpsc::Receiver<TradeDecision>) {
        info!("Execution engine: listening for trade decisions");
        while let Some(decision) = trade_rx.recv().await {
            let result = match &decision.decision_type {
                DecisionType::Buy              => self.execute_buy(decision).await,
                DecisionType::Sell             => self.execute_sell(decision, 1.0).await,
                DecisionType::PartialSell{pct} => self.execute_sell(decision, *pct).await,
                DecisionType::Skip             => Ok(()),
            };
            if let Err(e) = result { error!("Execution error: {e}"); }
        }
    }

    async fn execute_buy(&self, decision: TradeDecision) -> Result<()> {
        let mint       = &decision.signal.mint;
        let mint_short = &mint[..mint.len().min(8)];

        if self.config.bot.dry_run {
            info!("🧪 DRY BUY  {} | {:.4} SOL | ML {:.2}", mint_short, decision.buy_amount_sol, decision.filter_result.ensemble_score);
            let track = format!("{:?}", decision.strategy_track).to_lowercase();
            record_trade_entered(&track, decision.buy_amount_sol);
            self.db.log_dry_run_trade(&decision).await?;
            return Ok(());
        }

        if decision.entry_delay_seconds > 0 {
            debug!("Delaying {}s before entering {}", decision.entry_delay_seconds, mint_short);
            tokio::time::sleep(Duration::from_secs(decision.entry_delay_seconds as u64)).await;
            if self.has_price_dumped(mint, 0.15).await? {
                warn!("⛔ {} dumped during entry delay — skipping", mint_short);
                return Ok(());
            }
        }

        let on_chain     = decision.signal.on_chain.as_ref().context("No on-chain data in signal")?;
        let sol_lamports = (decision.buy_amount_sol * 1e9) as u64;
        let slippage_bps = decision.max_slippage_bps;
        let swap_ix      = self.build_buy_instruction(mint, on_chain, sol_lamports, slippage_bps).await?;

        let mut all_ixs = Vec::new();
        if matches!(on_chain.dex, DexType::RaydiumCPMM | DexType::RaydiumAMM) {
            all_ixs.append(&mut wrap_sol_instructions(&self.keypair.pubkey(), sol_lamports)?);
        }
        all_ixs.push(swap_ix);
        if matches!(on_chain.dex, DexType::RaydiumCPMM | DexType::RaydiumAMM) {
            all_ixs.push(unwrap_wsol_instruction(&self.keypair.pubkey())?);
        }

        let cu_price  = self.micro_lamports_per_cu().await;
        let final_ixs = [vec![set_compute_unit_limit(300_000), set_compute_unit_price(cu_price)], all_ixs].concat();

        let tip_lamports = self.calculate_tip(decision.buy_amount_sol).await;
        let blockhash    = self.rpc.get_latest_blockhash().await?;
        let bundle       = self.build_jito_bundle(final_ixs, tip_lamports, blockhash).await?;

        info!("🚀 BUY {} | {:.4} SOL | tip {} lamports", mint_short, decision.buy_amount_sol, tip_lamports);

        let results      = self.submit_bundle_parallel(&bundle).await;
        let any_accepted = results.iter().any(|r| r.is_ok());

        let _ = self.redis.record_bundle_submitted().await;
        if any_accepted { let _ = self.redis.record_bundle_accepted().await; }

        if self.config.bloxroute.enabled && any_accepted {
            let _ = self.submit_to_bloxroute(&bundle).await;
        }

        if !any_accepted {
            warn!("Bundle rejected by all engines for {} — retrying with higher slippage", mint_short);
            return self.retry_with_higher_slippage(decision).await;
        }

        let track = format!("{:?}", decision.strategy_track).to_lowercase();
        record_trade_entered(&track, decision.buy_amount_sol);

        let trade = Trade {
            id: decision.id, mint: mint.clone(),
            strategy_track: decision.strategy_track.clone(),
            status: TradeStatus::Submitted,
            entry_price_usd: on_chain.price_usd, entry_amount_sol: decision.buy_amount_sol,
            entry_tx: None, entered_at: Some(chrono::Utc::now()),
            exit_price_usd: None, exit_amount_sol: None,
            exit_tx: None, exited_at: None,
            pnl_sol: None, pnl_pct: None, peak_multiplier: None,
            jito_tip_lamports: Some(tip_lamports),
            filter_result: decision.filter_result.clone(),
            created_at: chrono::Utc::now(),
        };

        self.db.insert_trade(&trade).await?;
        self.redis.set_open_position(mint, &trade).await?;

        // FIX (Bug #4): Poll the actual Jito bundle status instead of
        // unconditionally marking Confirmed after 5 s.  A failed or stuck
        // bundle now correctly ends up as Failed, which keeps the circuit
        // breaker loss counter and P&L accurate.
        let bundle_id = results.iter().filter_map(|r| r.as_ref().ok()).next().cloned().unwrap_or_default();
        let engine    = self.config.jito.block_engines.first().cloned().unwrap_or_default();
        let db_clone  = self.db.clone();
        let redis_clone = self.redis.clone();
        let trade_id  = trade.id;
        let mint_for_task = mint.clone();
        let http_clone = self.http.clone();

        tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
            loop {
                tokio::time::sleep(Duration::from_secs(3)).await;
                if tokio::time::Instant::now() >= deadline {
                    // Timeout — treat as failed to avoid phantom open positions.
                    warn!("Bundle confirmation timeout for trade {} — marking failed", trade_id);
                    let _ = db_clone.update_trade_failed(trade_id).await;
                    let _ = redis_clone.remove_open_position(&mint_for_task).await;
                    return;
                }
                match poll_bundle_status(&http_clone, &engine, &bundle_id).await {
                    Ok(status) if status == "landed" => {
                        info!("✅ Bundle landed for trade {}", trade_id);
                        let _ = db_clone.update_trade_status(trade_id, TradeStatus::Confirmed).await;
                        return;
                    }
                    Ok(status) if status == "failed" || status == "invalid" => {
                        warn!("Bundle {} — marking trade {} as failed", status, trade_id);
                        let _ = db_clone.update_trade_failed(trade_id).await;
                        let _ = redis_clone.remove_open_position(&mint_for_task).await;
                        return;
                    }
                    Ok(_)  => {} // still pending — keep polling
                    Err(e) => { debug!("Bundle status poll error: {}", e); }
                }
            }
        });

        Ok(())
    }

    async fn execute_sell(&self, decision: TradeDecision, sell_pct: f64) -> Result<()> {
        let mint       = &decision.signal.mint;
        let mint_short = &mint[..mint.len().min(8)];

        if self.config.bot.dry_run {
            info!("🧪 DRY SELL {:.0}% of {}", sell_pct * 100.0, mint_short);
            return Ok(());
        }

        let token_balance = self.rpc.get_token_balance(&self.keypair.pubkey().to_string(), mint).await.unwrap_or(0);
        if token_balance == 0 { warn!("No token balance to sell for {}", mint_short); return Ok(()); }

        let sell_amount = (token_balance as f64 * sell_pct) as u64;
        if sell_amount == 0 { return Ok(()); }

        let on_chain = decision.signal.on_chain.as_ref().context("No on-chain data for sell")?;
        let sell_ix  = self.build_sell_instruction(mint, on_chain, sell_amount, decision.max_slippage_bps).await?;

        let cu_price = self.micro_lamports_per_cu().await;
        let mut all_ixs = vec![set_compute_unit_limit(250_000), set_compute_unit_price(cu_price), sell_ix];
        if matches!(on_chain.dex, DexType::RaydiumCPMM | DexType::RaydiumAMM) {
            all_ixs.push(unwrap_wsol_instruction(&self.keypair.pubkey())?);
        }

        let tip_lamports = self.config.jito.tip_min_lamports * 2;
        let blockhash    = self.rpc.get_latest_blockhash().await?;
        let bundle       = self.build_jito_bundle(all_ixs, tip_lamports, blockhash).await?;
        let results      = self.submit_bundle_parallel(&bundle).await;

        if results.iter().any(|r| r.is_ok()) {
            info!("🔴 SELL {:.0}% of {} submitted", sell_pct * 100.0, mint_short);
            // Position is removed here (after sell confirmed) — NOT in fire_full_exit.
            // See Bug #5 fix in strategy/mod.rs.
            if sell_pct >= 1.0 {
                self.redis.remove_open_position(mint).await?;
            }
        } else {
            error!("Sell bundle failed for {} — will retry next position-manager tick", mint_short);
        }
        Ok(())
    }

    async fn build_buy_instruction(&self, mint: &str, on_chain: &OnChainData, sol_lamports: u64, slippage_bps: u32) -> Result<solana_sdk::instruction::Instruction> {
        use std::str::FromStr;
        use solana_sdk::pubkey::Pubkey;
        let mint_pubkey = Pubkey::from_str(mint)?;
        let user = self.keypair.pubkey();
        match on_chain.dex {
            DexType::PumpFun | DexType::PumpSwap => {
                let (bonding_curve, _) = derive_pump_fun_bonding_curve(&mint_pubkey);
                let bc_ata = spl_associated_token_account::get_associated_token_address(&bonding_curve, &mint_pubkey);
                build_pump_fun_buy(&mint_pubkey, &bonding_curve, &bc_ata, &user, sol_lamports, slippage_bps,
                    on_chain.price_usd.max(1e-10), self.get_sol_price_usd().await)
            }
            DexType::RaydiumCPMM => {
                let pool_state = Pubkey::from_str(&on_chain.pool_address).unwrap_or_default();
                let (token0_vault, token1_vault) = self.fetch_raydium_cpmm_vaults(&on_chain.pool_address).await?;
                let wsol_mint = Pubkey::from_str(WSOL_MINT)?;
                let pool = CpmmPoolAccounts {
                    pool_id: pool_state, pool_authority: Pubkey::from_str(RAYDIUM_CPMM_AUTHORITY)?,
                    pool_state, token0_mint: wsol_mint, token1_mint: mint_pubkey,
                    token0_vault, token1_vault, lp_mint: Pubkey::default(), observation_key: Pubkey::default(),
                };
                let min_out = apply_slippage(self.estimate_token_out(sol_lamports, on_chain), slippage_bps);
                build_raydium_cpmm_swap(&pool, &user, sol_lamports, min_out, true)
            }
            DexType::RaydiumAMM => {
                let pool    = self.fetch_raydium_amm_accounts(&on_chain.pool_address).await?;
                let min_out = apply_slippage(self.estimate_token_out(sol_lamports, on_chain), slippage_bps);
                build_raydium_amm_swap(&pool, &user, sol_lamports, min_out, true)
            }
            _ => self.build_jupiter_swap_ix(mint, sol_lamports, slippage_bps, true).await,
        }
    }

    async fn build_sell_instruction(&self, mint: &str, on_chain: &OnChainData, token_amount: u64, slippage_bps: u32) -> Result<solana_sdk::instruction::Instruction> {
        use std::str::FromStr;
        use solana_sdk::pubkey::Pubkey;
        let mint_pubkey = Pubkey::from_str(mint)?;
        let user = self.keypair.pubkey();
        match on_chain.dex {
            DexType::PumpFun | DexType::PumpSwap => {
                let (bonding_curve, _) = derive_pump_fun_bonding_curve(&mint_pubkey);
                let bc_ata = spl_associated_token_account::get_associated_token_address(&bonding_curve, &mint_pubkey);
                build_pump_fun_sell(&mint_pubkey, &bonding_curve, &bc_ata, &user, token_amount, slippage_bps,
                    on_chain.price_usd.max(1e-10), self.get_sol_price_usd().await)
            }
            DexType::RaydiumCPMM => {
                let pool_state = Pubkey::from_str(&on_chain.pool_address).unwrap_or_default();
                let (token0_vault, token1_vault) = self.fetch_raydium_cpmm_vaults(&on_chain.pool_address).await?;
                let wsol_mint = Pubkey::from_str(WSOL_MINT)?;
                let pool = CpmmPoolAccounts {
                    pool_id: pool_state, pool_authority: Pubkey::from_str(RAYDIUM_CPMM_AUTHORITY)?,
                    pool_state, token0_mint: wsol_mint, token1_mint: mint_pubkey,
                    token0_vault, token1_vault, lp_mint: Pubkey::default(), observation_key: Pubkey::default(),
                };
                let min_sol_out = apply_slippage(self.estimate_sol_out(token_amount, on_chain), slippage_bps);
                build_raydium_cpmm_swap(&pool, &user, token_amount, min_sol_out, false)
            }
            _ => self.build_jupiter_swap_ix(mint, token_amount, slippage_bps, false).await,
        }
    }

    async fn build_jupiter_swap_ix(&self, mint: &str, amount: u64, slippage_bps: u32, is_buy: bool) -> Result<solana_sdk::instruction::Instruction> {
        use std::str::FromStr;
        let (input_mint, output_mint) = if is_buy { (WSOL_MINT, mint) } else { (mint, WSOL_MINT) };
        let quote_url = format!(
            "https://quote-api.jup.ag/v6/quote?inputMint={}&outputMint={}&amount={}&slippageBps={}",
            input_mint, output_mint, amount, slippage_bps,
        );
        let quote: serde_json::Value = self.http.get(&quote_url).send().await?.json().await?;
        let swap_body = serde_json::json!({
            "quoteResponse": quote,
            "userPublicKey": self.keypair.pubkey().to_string(),
            "wrapAndUnwrapSol": true,
            "skipUserAccountsRpcCalls": true,
            "dynamicComputeUnitLimit": false,
        });
        let swap_resp: serde_json::Value = self.http
            .post("https://quote-api.jup.ag/v6/swap-instructions")
            .json(&swap_body).send().await?.json().await?;
        let tx_b64   = swap_resp["swapTransaction"].as_str().context("No swapTransaction in Jupiter response")?;
        let tx_bytes = base64::engine::general_purpose::STANDARD.decode(tx_b64)?;
        let tx: solana_sdk::transaction::VersionedTransaction = bincode::deserialize(&tx_bytes)?;
        let message      = tx.message;
        let account_keys: Vec<solana_sdk::pubkey::Pubkey> = message.static_account_keys().to_vec();
        let compute_prog = solana_sdk::pubkey::Pubkey::from_str(COMPUTE_BUDGET_PROGRAM_ID)?;
        for ix in message.instructions() {
            let prog = account_keys.get(ix.program_id_index as usize).copied().unwrap_or_default();
            if prog != compute_prog {
                let accounts: Vec<solana_sdk::instruction::AccountMeta> = ix.accounts.iter()
                    .map(|&idx| solana_sdk::instruction::AccountMeta {
                        pubkey: account_keys[idx as usize],
                        is_signer: false,
                        is_writable: true,
                    }).collect();
                return Ok(solana_sdk::instruction::Instruction { program_id: prog, accounts, data: ix.data.clone() });
            }
        }
        anyhow::bail!("Could not extract swap instruction from Jupiter transaction")
    }

    async fn build_jito_bundle(&self, instructions: Vec<solana_sdk::instruction::Instruction>, tip_lamports: u64, blockhash: solana_sdk::hash::Hash) -> Result<serde_json::Value> {
        use std::str::FromStr;
        let tip_ix = solana_sdk::system_instruction::transfer(
            &self.keypair.pubkey(),
            &solana_sdk::pubkey::Pubkey::from_str(&self.config.jito.tip_account)?,
            tip_lamports,
        );
        let all_ixs = [instructions, vec![tip_ix]].concat();
        let tx = solana_sdk::transaction::Transaction::new_signed_with_payer(
            &all_ixs, Some(&self.keypair.pubkey()), &[&self.keypair], blockhash,
        );
        let tx_bytes = bincode::serialize(&tx)?;
        let tx_b64   = base64::engine::general_purpose::STANDARD.encode(&tx_bytes);
        Ok(serde_json::json!({"jsonrpc":"2.0","id":1,"method":"sendBundle","params":[[tx_b64]]}))
    }

    async fn submit_bundle_parallel(&self, bundle: &serde_json::Value) -> Vec<Result<String>> {
        let futs: Vec<_> = self.config.jito.block_engines.iter().map(|ep| {
            let client = self.http.clone();
            let bundle = bundle.clone();
            let ep = ep.clone();
            async move {
                let resp: serde_json::Value = client.post(&ep)
                    .json(&bundle).timeout(Duration::from_secs(4)).send().await?.json().await?;
                if let Some(result) = resp.get("result") {
                    Ok(result.as_str().unwrap_or("").to_string())
                } else {
                    Err(anyhow::anyhow!("Jito error: {}", resp.get("error").unwrap_or(&serde_json::Value::Null)))
                }
            }
        }).collect();
        futures::future::join_all(futs).await
    }

    async fn submit_to_bloxroute(&self, bundle: &serde_json::Value) -> Result<()> {
        let cfg = &self.config.bloxroute;
        self.http.post(&cfg.endpoint)
            .header("Authorization", &cfg.auth_header)
            .json(bundle).timeout(Duration::from_secs(4)).send().await?;
        Ok(())
    }

    async fn retry_with_higher_slippage(&self, mut decision: TradeDecision) -> Result<()> {
        let max_slippage = self.config.execution.slippage_max_bps;
        let increment    = self.config.execution.slippage_increment_bps;
        let mint_short   = &decision.signal.mint[..decision.signal.mint.len().min(8)];
        for attempt in 1..=self.config.execution.max_retries {
            let new_slippage = (decision.max_slippage_bps + increment * attempt).min(max_slippage);
            decision.max_slippage_bps = new_slippage;
            tokio::time::sleep(Duration::from_millis(self.config.execution.retry_delay_ms)).await;
            if let Some(on_chain) = decision.signal.on_chain.as_ref() {
                let sol_lamports = (decision.buy_amount_sol * 1e9) as u64;
                if let Ok(swap_ix) = self.build_buy_instruction(&decision.signal.mint, on_chain, sol_lamports, new_slippage).await {
                    let blockhash = self.rpc.get_latest_blockhash().await?;
                    let tip       = self.calculate_tip(decision.buy_amount_sol).await;
                    let cu_price  = self.micro_lamports_per_cu().await;
                    let bundle    = self.build_jito_bundle(
                        vec![set_compute_unit_limit(300_000), set_compute_unit_price(cu_price), swap_ix],
                        tip, blockhash,
                    ).await?;
                    let results = self.submit_bundle_parallel(&bundle).await;
                    if results.iter().any(|r| r.is_ok()) {
                        info!("✅ Retry {} succeeded for {}", attempt, mint_short);
                        return Ok(());
                    }
                }
            }
        }
        warn!("All retries exhausted for {}", mint_short);
        Ok(())
    }

    async fn calculate_tip(&self, buy_sol: f64) -> u64 {
        let cfg = &self.config.jito;
        let expected_profit = (buy_sol * 1e9) as u64;
        let base_tip = (expected_profit as f64 * cfg.tip_profit_share) as u64;
        let multiplier = self.redis.get_network_congestion_multiplier().await.unwrap_or(1.0);
        ((base_tip as f64 * multiplier) as u64).clamp(cfg.tip_min_lamports, cfg.tip_max_lamports)
    }

    async fn micro_lamports_per_cu(&self) -> u64 {
        let multiplier = self.redis.get_network_congestion_multiplier().await.unwrap_or(1.0);
        (5_000.0 * multiplier) as u64
    }

    async fn has_price_dumped(&self, mint: &str, threshold: f64) -> Result<bool> {
        let current = match self.redis.get_cached_price(mint).await {
            Some(p) => p,
            None    => return Ok(false),
        };
        let positions = self.redis.get_all_open_positions().await?;
        if let Some(pos) = positions.iter().find(|p| p.mint == mint) {
            if pos.entry_price_usd > 0.0 {
                return Ok(1.0 - (current / pos.entry_price_usd) >= threshold);
            }
        }
        Ok(false)
    }

    async fn get_sol_price_usd(&self) -> f64 {
        self.redis.get_cached_price("SOL").await.unwrap_or(160.0)
    }

    fn estimate_token_out(&self, sol_lamports: u64, on_chain: &OnChainData) -> u64 {
        if on_chain.price_usd <= 0.0 { return 0; }
        let sol_usd = sol_lamports as f64 / 1e9 * 160.0;
        ((sol_usd / on_chain.price_usd) * 1e6) as u64
    }

    fn estimate_sol_out(&self, token_amount: u64, on_chain: &OnChainData) -> u64 {
        let token_usd = token_amount as f64 / 1e6 * on_chain.price_usd;
        (token_usd / 160.0 * 1e9) as u64
    }

    async fn fetch_raydium_cpmm_vaults(&self, pool_address: &str) -> Result<(solana_sdk::pubkey::Pubkey, solana_sdk::pubkey::Pubkey)> {
        use std::str::FromStr;
        let accounts = self.rpc.get_multiple_accounts(&[pool_address]).await?;
        if let Some(Some(account)) = accounts.first() {
            if let Some(data_b64) = account["data"][0].as_str() {
                let data = base64::engine::general_purpose::STANDARD.decode(data_b64)?;
                if data.len() >= 232 {
                    let v0 = solana_sdk::pubkey::Pubkey::from(<[u8; 32]>::try_from(&data[168..200])?);
                    let v1 = solana_sdk::pubkey::Pubkey::from(<[u8; 32]>::try_from(&data[200..232])?);
                    return Ok((v0, v1));
                }
            }
        }
        let pool_pk   = solana_sdk::pubkey::Pubkey::from_str(pool_address)?;
        let wsol_mint = solana_sdk::pubkey::Pubkey::from_str(WSOL_MINT)?;
        let cpmm      = solana_sdk::pubkey::Pubkey::from_str(RAYDIUM_CPMM_PROGRAM_ID)?;
        let (v0, _)   = solana_sdk::pubkey::Pubkey::find_program_address(&[b"pool_vault", pool_pk.as_ref(), wsol_mint.as_ref()], &cpmm);
        let (v1, _)   = solana_sdk::pubkey::Pubkey::find_program_address(&[b"pool_vault", pool_pk.as_ref(), pool_pk.as_ref()], &cpmm);
        Ok((v0, v1))
    }

    async fn fetch_raydium_amm_accounts(&self, pool_address: &str) -> Result<AmmV4PoolAccounts> {
        use std::str::FromStr;
        let accounts = self.rpc.get_multiple_accounts(&[pool_address]).await?;
        if let Some(Some(account)) = accounts.first() {
            if let Some(data_b64) = account["data"][0].as_str() {
                let data = base64::engine::general_purpose::STANDARD.decode(data_b64)?;
                if data.len() >= 288 {
                    let parse_pk = |slice: &[u8]| solana_sdk::pubkey::Pubkey::from(<[u8; 32]>::try_from(slice).unwrap_or([0u8; 32]));
                    let amm_pk   = solana_sdk::pubkey::Pubkey::from_str(pool_address)?;
                    let amm_prog = solana_sdk::pubkey::Pubkey::from_str(RAYDIUM_AMM_PROGRAM_ID)?;
                    let (authority, _) = solana_sdk::pubkey::Pubkey::find_program_address(&[b"amm authority"], &amm_prog);
                    return Ok(AmmV4PoolAccounts {
                        amm_id: amm_pk, amm_authority: authority,
                        amm_open_orders:         parse_pk(&data[32..64]),
                        amm_target_orders:       parse_pk(&data[64..96]),
                        pool_coin_token_account: parse_pk(&data[96..128]),
                        pool_pc_token_account:   parse_pk(&data[128..160]),
                        serum_program_id:        parse_pk(&data[256..288]),
                        serum_market:            parse_pk(&data[224..256]),
                        serum_bids:              solana_sdk::pubkey::Pubkey::default(),
                        serum_asks:              solana_sdk::pubkey::Pubkey::default(),
                        serum_event_queue:       solana_sdk::pubkey::Pubkey::default(),
                        serum_coin_vault:        solana_sdk::pubkey::Pubkey::default(),
                        serum_pc_vault:          solana_sdk::pubkey::Pubkey::default(),
                        serum_vault_signer:      solana_sdk::pubkey::Pubkey::default(),
                        base_mint:               solana_sdk::pubkey::Pubkey::default(),
                        quote_mint:              solana_sdk::pubkey::Pubkey::from_str(WSOL_MINT)?,
                    });
                }
            }
        }
        anyhow::bail!("Cannot decode Raydium AMM pool state for {}", pool_address)
    }
}

// ── Jito bundle status polling ────────────────────────────────────────────────
//
// FIX (Bug #4): polls getBundleStatuses on the Jito block-engine so we know
// whether a bundle actually landed before updating the trade record.

async fn poll_bundle_status(http: &reqwest::Client, engine_url: &str, bundle_id: &str) -> Result<String> {
    if bundle_id.is_empty() { return Ok("pending".into()); }
    // Strip /bundles suffix if present to get the base URL
    let base = engine_url.trim_end_matches("/api/v1/bundles");
    let url  = format!("{}/api/v1/bundles", base);
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBundleStatuses",
        "params": [[bundle_id]],
    });
    let resp: serde_json::Value = http.post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(5))
        .send().await?.json().await?;

    let status = resp["result"]["value"][0]["confirmation_status"]
        .as_str()
        .or_else(|| resp["result"]["value"][0]["status"].as_str())
        .unwrap_or("pending");

    // Jito returns: "invalid", "pending", "failed", "landed" / "confirmed"
    match status {
        "confirmed" | "finalized" => Ok("landed".into()),
        "landed"  => Ok("landed".into()),
        "failed"  => Ok("failed".into()),
        "invalid" => Ok("invalid".into()),
        _         => Ok("pending".into()),
    }
}
