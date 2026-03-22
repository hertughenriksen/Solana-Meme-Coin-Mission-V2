/// scanner/instruction_parser.rs
/// ================================
/// Decodes Yellowstone gRPC transaction bytes into structured events.
/// Covers Pump.fun launches/swaps and Raydium CPMM launches/swaps.

use std::collections::HashSet;

const PUMP_CREATE_DISC:     [u8; 8] = [24,  30, 200,  40,   5,  28,   7, 119];
const PUMP_BUY_DISC:        [u8; 8] = [102,  6,  61,  18,   1, 218, 235, 234];
const PUMP_SELL_DISC:       [u8; 8] = [ 51, 230, 133, 164,   1, 127, 131, 173];
const RAYDIUM_CPMM_INIT:    [u8; 8] = [175, 175, 109,  31,  13, 152, 155, 237];
const RAYDIUM_CPMM_SWAP_IN: [u8; 8] = [143, 190,  90, 218, 196,  30,  51, 222];

pub const PUMP_FUN_PROG:     &str = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P";
pub const RAYDIUM_CPMM_PROG: &str = "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C";
pub const RAYDIUM_AMM_PROG:  &str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8";
pub const METEORA_PROG:      &str = "Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EkAW7Es";
pub const WSOL_MINT:         &str = "So11111111111111111111111111111111111111112";

pub struct DecodedTransaction {
    pub signature: String,
    pub account_keys: Vec<String>,
    pub instructions: Vec<RawInstruction>,
    pub inner_instructions: Vec<InnerGroup>,
    pub pre_token_balances: Vec<TokenBalance>,
    pub post_token_balances: Vec<TokenBalance>,
    pub pre_sol_balances: Vec<u64>,
    pub post_sol_balances: Vec<u64>,
}

pub struct RawInstruction {
    pub program_id_index: usize,
    pub account_indices: Vec<usize>,
    pub data: Vec<u8>,
}

pub struct InnerGroup {
    pub index: u32,
    pub instructions: Vec<RawInstruction>,
}

#[derive(Debug, Clone)]
pub struct TokenBalance {
    pub account_index: u32,
    pub mint: String,
    pub owner: String,
    pub amount: u64,
}

#[derive(Debug, Clone)]
pub struct LaunchInfo {
    pub mint: String,
    pub pool_address: String,
    pub dex: DexKind,
    pub deployer: String,
    pub initial_sol_lamports: u64,
}

#[derive(Debug, Clone)]
pub struct SwapInfo {
    pub token_mint: String,
    pub pool_address: String,
    pub dex: DexKind,
    pub is_buy: bool,
    pub sol_amount: f64,
    pub token_amount: u64,
    pub price_usd: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DexKind { PumpFun, PumpSwap, RaydiumCpmm, RaydiumAmm, Meteora }

#[cfg(feature = "yellowstone")]
pub fn decode_transaction(
    tx_info: &yellowstone_grpc_proto::geyser::SubscribeUpdateTransactionInfo,
    meta: &yellowstone_grpc_proto::geyser::TransactionStatusMeta,
) -> Option<DecodedTransaction> {
    let tx  = tx_info.transaction.as_ref()?;
    let msg = tx.message.as_ref()?;

    let account_keys: Vec<String> = msg.account_keys.iter()
        .map(|k| bs58::encode(k).into_string())
        .collect();

    let decode_raw = |ix: &yellowstone_grpc_proto::geyser::CompiledInstruction| RawInstruction {
        program_id_index: ix.program_id_index as usize,
        account_indices:  ix.accounts.iter().map(|&a| a as usize).collect(),
        data:             ix.data.clone(),
    };

    let instructions: Vec<RawInstruction> = msg.instructions.iter().map(decode_raw).collect();

    let inner_instructions = meta.inner_instructions.iter().map(|ig| InnerGroup {
        index: ig.index,
        instructions: ig.instructions.iter().map(decode_raw).collect(),
    }).collect();

    let decode_bal = |b: &yellowstone_grpc_proto::geyser::TokenBalance| TokenBalance {
        account_index: b.account_index,
        mint:  b.mint.clone(),
        owner: b.owner.clone(),
        amount: b.ui_token_amount.as_ref()
            .and_then(|a| a.amount.parse::<u64>().ok())
            .unwrap_or(0),
    };

    let signature = tx.signatures.first()
        .map(|s| bs58::encode(s).into_string())
        .unwrap_or_default();

    Some(DecodedTransaction {
        signature,
        account_keys,
        instructions,
        inner_instructions,
        pre_token_balances:  meta.pre_token_balances.iter().map(decode_bal).collect(),
        post_token_balances: meta.post_token_balances.iter().map(decode_bal).collect(),
        pre_sol_balances:    meta.pre_balances.clone(),
        post_sol_balances:   meta.post_balances.clone(),
    })
}

pub fn resolve_program(ix: &RawInstruction, keys: &[String]) -> Option<String> {
    keys.get(ix.program_id_index).cloned()
}

pub fn resolve_accounts(ix: &RawInstruction, keys: &[String]) -> Vec<String> {
    ix.account_indices.iter().filter_map(|&i| keys.get(i).cloned()).collect()
}

pub fn all_resolved_instructions(tx: &DecodedTransaction) -> Vec<(String, Vec<String>, Vec<u8>)> {
    let mut out = Vec::new();
    for raw in &tx.instructions {
        if let Some(prog) = resolve_program(raw, &tx.account_keys) {
            out.push((prog, resolve_accounts(raw, &tx.account_keys), raw.data.clone()));
        }
    }
    for group in &tx.inner_instructions {
        for raw in &group.instructions {
            if let Some(prog) = resolve_program(raw, &tx.account_keys) {
                out.push((prog, resolve_accounts(raw, &tx.account_keys), raw.data.clone()));
            }
        }
    }
    out
}

pub fn get_signers(tx: &DecodedTransaction) -> Vec<String> {
    tx.account_keys.iter().take(1).cloned().collect()
}

pub fn involved_programs<'a>(tx: &'a DecodedTransaction) -> HashSet<&'a str> {
    tx.instructions.iter()
        .filter_map(|ix| tx.account_keys.get(ix.program_id_index).map(|s| s.as_str()))
        .collect()
}

pub fn parse_pump_fun_launch(tx: &DecodedTransaction) -> Option<LaunchInfo> {
    for raw in &tx.instructions {
        let prog = resolve_program(raw, &tx.account_keys)?;
        if prog != PUMP_FUN_PROG { continue; }
        if raw.data.len() < 8 || raw.data[..8] != PUMP_CREATE_DISC { continue; }
        let accs          = resolve_accounts(raw, &tx.account_keys);
        let mint          = accs.get(0)?.clone();
        let bonding_curve = accs.get(2)?.clone();
        let deployer      = accs.get(7)?.clone();
        let deployer_idx  = tx.account_keys.iter().position(|k| *k == deployer)?;
        let sol_spent = tx.pre_sol_balances.get(deployer_idx)
            .zip(tx.post_sol_balances.get(deployer_idx))
            .map(|(pre, post)| pre.saturating_sub(*post))
            .unwrap_or(0);
        return Some(LaunchInfo { mint, pool_address: bonding_curve, dex: DexKind::PumpFun, deployer, initial_sol_lamports: sol_spent });
    }
    None
}

pub fn parse_raydium_cpmm_launch(tx: &DecodedTransaction) -> Option<LaunchInfo> {
    for raw in &tx.instructions {
        let prog = resolve_program(raw, &tx.account_keys)?;
        if prog != RAYDIUM_CPMM_PROG { continue; }
        if raw.data.len() < 8 || raw.data[..8] != RAYDIUM_CPMM_INIT { continue; }
        let accs        = resolve_accounts(raw, &tx.account_keys);
        let deployer    = accs.get(0)?.clone();
        let pool_state  = accs.get(3)?.clone();
        let token0_mint = accs.get(4)?.clone();
        let token1_mint = accs.get(5)?.clone();
        let new_token   = if token0_mint == WSOL_MINT { token1_mint } else { token0_mint };
        let deployer_idx = tx.account_keys.iter().position(|k| *k == deployer)?;
        let sol_spent = tx.pre_sol_balances.get(deployer_idx)
            .zip(tx.post_sol_balances.get(deployer_idx))
            .map(|(pre, post)| pre.saturating_sub(*post))
            .unwrap_or(0);
        return Some(LaunchInfo { mint: new_token, pool_address: pool_state, dex: DexKind::RaydiumCpmm, deployer, initial_sol_lamports: sol_spent });
    }
    None
}

pub fn parse_pump_fun_swap(tx: &DecodedTransaction) -> Option<SwapInfo> {
    for raw in &tx.instructions {
        let prog = resolve_program(raw, &tx.account_keys)?;
        if prog != PUMP_FUN_PROG { continue; }
        if raw.data.len() < 24 { continue; }
        let is_buy  = raw.data[..8] == PUMP_BUY_DISC;
        let is_sell = raw.data[..8] == PUMP_SELL_DISC;
        if !is_buy && !is_sell { continue; }
        let accs      = resolve_accounts(raw, &tx.account_keys);
        let mint      = accs.get(2)?.clone();
        let pool      = accs.get(3)?.clone();
        let amount    = u64::from_le_bytes(raw.data[8..16].try_into().ok()?);
        let sol_limit = u64::from_le_bytes(raw.data[16..24].try_into().ok()?);
        return Some(SwapInfo { token_mint: mint, pool_address: pool, dex: DexKind::PumpFun, is_buy, sol_amount: sol_limit as f64 / 1e9, token_amount: amount, price_usd: 0.0 });
    }
    None
}

pub fn parse_raydium_cpmm_swap(tx: &DecodedTransaction) -> Option<SwapInfo> {
    for raw in &tx.instructions {
        let prog = resolve_program(raw, &tx.account_keys)?;
        if prog != RAYDIUM_CPMM_PROG { continue; }
        if raw.data.len() < 24 { continue; }
        if raw.data[..8] != RAYDIUM_CPMM_SWAP_IN { continue; }
        let accs        = resolve_accounts(raw, &tx.account_keys);
        let pool_state  = accs.get(3)?.clone();
        let input_mint  = accs.get(10)?.clone();
        let output_mint = accs.get(11)?.clone();
        let is_buy      = input_mint == WSOL_MINT;
        let token       = if is_buy { output_mint } else { input_mint };
        let amount_in   = u64::from_le_bytes(raw.data[8..16].try_into().ok()?);
        return Some(SwapInfo { token_mint: token, pool_address: pool_state, dex: DexKind::RaydiumCpmm, is_buy, sol_amount: amount_in as f64 / 1e9, token_amount: u64::from_le_bytes(raw.data[16..24].try_into().ok()?), price_usd: 0.0 });
    }
    None
}

pub fn is_sniper_bundle(tx: &DecodedTransaction, mint: &str) -> bool {
    let inflows = tx.post_token_balances.iter()
        .filter(|b| b.mint == mint)
        .filter(|b| {
            let pre = tx.pre_token_balances.iter()
                .find(|p| p.account_index == b.account_index && p.mint == mint)
                .map(|p| p.amount).unwrap_or(0);
            b.amount > pre
        })
        .count();
    inflows >= 3
}
