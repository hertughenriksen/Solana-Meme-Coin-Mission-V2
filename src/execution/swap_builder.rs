use anyhow::Result;
use borsh::{BorshDeserialize, BorshSerialize};
use solana_sdk::{
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
    system_program,
};
use spl_associated_token_account::get_associated_token_address;
use std::str::FromStr;

pub const PUMP_FUN_PROGRAM_ID:         &str = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P";
pub const PUMP_SWAP_PROGRAM_ID:        &str = "pSwapMdSai8tjrEXcxFeQth87xC4rRsa4Ka5Sw4UDCZ";
pub const RAYDIUM_CPMM_PROGRAM_ID:     &str = "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C";
pub const RAYDIUM_AMM_PROGRAM_ID:      &str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8";
pub const TOKEN_PROGRAM_ID:            &str = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA";
pub const TOKEN_2022_PROGRAM_ID:       &str = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb";
pub const ASSOCIATED_TOKEN_PROGRAM_ID: &str = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bPq";
pub const RENT_PROGRAM_ID:             &str = "SysvarRent111111111111111111111111111111111";
pub const COMPUTE_BUDGET_PROGRAM_ID:   &str = "ComputeBudget111111111111111111111111111111";
pub const WSOL_MINT:                   &str = "So11111111111111111111111111111111111111112";
pub const PUMP_FUN_GLOBAL:             &str = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5zP9QkVe5m5EvF";
pub const PUMP_FUN_FEE_RECIPIENT:      &str = "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM";
pub const PUMP_FUN_EVENT_AUTHORITY:    &str = "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1";
pub const RAYDIUM_CPMM_AUTHORITY:      &str = "GpMZbSM2GgvTKHJirzeGfMFoaZ8UR2X7F4v8vHTvxFbR";

// ── Compute budget ────────────────────────────────────────────────────────────

pub fn set_compute_unit_limit(units: u32) -> Instruction {
    Instruction {
        program_id: Pubkey::from_str(COMPUTE_BUDGET_PROGRAM_ID).unwrap(),
        accounts: vec![],
        data: { let mut d = vec![0x02u8]; d.extend_from_slice(&units.to_le_bytes()); d },
    }
}

pub fn set_compute_unit_price(micro_lamports: u64) -> Instruction {
    Instruction {
        program_id: Pubkey::from_str(COMPUTE_BUDGET_PROGRAM_ID).unwrap(),
        accounts: vec![],
        data: { let mut d = vec![0x03u8]; d.extend_from_slice(&micro_lamports.to_le_bytes()); d },
    }
}

// ── Pump.fun ──────────────────────────────────────────────────────────────────

#[derive(BorshSerialize, BorshDeserialize)]
pub struct PumpFunBuyArgs  { pub amount: u64, pub max_sol_cost: u64 }
#[derive(BorshSerialize, BorshDeserialize)]
pub struct PumpFunSellArgs { pub amount: u64, pub min_sol_output: u64 }

pub fn build_pump_fun_buy(
    mint: &Pubkey, bonding_curve: &Pubkey, bc_ata: &Pubkey,
    user: &Pubkey, sol_lamports: u64, slippage_bps: u32,
    token_price_usd: f64, sol_price_usd: f64,
) -> Result<Instruction> {
    let user_ata  = get_associated_token_address(user, mint);
    let sol_usd   = sol_lamports as f64 / 1e9 * sol_price_usd;
    let token_amt = ((sol_usd / token_price_usd.max(1e-12)) * 1e6) as u64;
    let max_sol   = sol_lamports + (sol_lamports * slippage_bps as u64) / 10_000;

    let args = PumpFunBuyArgs { amount: token_amt, max_sol_cost: max_sol };
    let mut data = vec![102u8, 6, 61, 18, 1, 218, 235, 234]; // buy discriminator
    data.extend_from_slice(&borsh::to_vec(&args)?);

    Ok(Instruction {
        program_id: Pubkey::from_str(PUMP_FUN_PROGRAM_ID)?,
        accounts: vec![
            AccountMeta::new_readonly(Pubkey::from_str(PUMP_FUN_GLOBAL)?,         false),
            AccountMeta::new(        Pubkey::from_str(PUMP_FUN_FEE_RECIPIENT)?,   false),
            AccountMeta::new_readonly(*mint,                                       false),
            AccountMeta::new(        *bonding_curve,                               false),
            AccountMeta::new(        *bc_ata,                                      false),
            AccountMeta::new(        user_ata,                                     false),
            AccountMeta::new(        *user,                                        true),
            AccountMeta::new_readonly(system_program::id(),                        false),
            AccountMeta::new_readonly(Pubkey::from_str(TOKEN_PROGRAM_ID)?,         false),
            AccountMeta::new_readonly(Pubkey::from_str(RENT_PROGRAM_ID)?,          false),
            AccountMeta::new_readonly(Pubkey::from_str(PUMP_FUN_EVENT_AUTHORITY)?, false),
            AccountMeta::new_readonly(Pubkey::from_str(PUMP_FUN_PROGRAM_ID)?,      false),
        ],
        data,
    })
}

pub fn build_pump_fun_sell(
    mint: &Pubkey, bonding_curve: &Pubkey, bc_ata: &Pubkey,
    user: &Pubkey, token_amount: u64, slippage_bps: u32,
    token_price_usd: f64, sol_price_usd: f64,
) -> Result<Instruction> {
    let user_ata     = get_associated_token_address(user, mint);
    let token_usd    = token_amount as f64 / 1e6 * token_price_usd;
    let expected_sol = (token_usd / sol_price_usd.max(1e-6) * 1e9) as u64;
    let min_sol      = expected_sol.saturating_sub((expected_sol * slippage_bps as u64) / 10_000);

    let args = PumpFunSellArgs { amount: token_amount, min_sol_output: min_sol };
    let mut data = vec![51u8, 230, 133, 164, 1, 127, 131, 173]; // sell discriminator
    data.extend_from_slice(&borsh::to_vec(&args)?);

    Ok(Instruction {
        program_id: Pubkey::from_str(PUMP_FUN_PROGRAM_ID)?,
        accounts: vec![
            AccountMeta::new_readonly(Pubkey::from_str(PUMP_FUN_GLOBAL)?,             false),
            AccountMeta::new(        Pubkey::from_str(PUMP_FUN_FEE_RECIPIENT)?,        false),
            AccountMeta::new_readonly(*mint,                                            false),
            AccountMeta::new(        *bonding_curve,                                   false),
            AccountMeta::new(        *bc_ata,                                          false),
            AccountMeta::new(        user_ata,                                         false),
            AccountMeta::new(        *user,                                            true),
            AccountMeta::new_readonly(system_program::id(),                            false),
            AccountMeta::new_readonly(Pubkey::from_str(ASSOCIATED_TOKEN_PROGRAM_ID)?, false),
            AccountMeta::new_readonly(Pubkey::from_str(TOKEN_PROGRAM_ID)?,             false),
            AccountMeta::new_readonly(Pubkey::from_str(PUMP_FUN_EVENT_AUTHORITY)?,     false),
            AccountMeta::new_readonly(Pubkey::from_str(PUMP_FUN_PROGRAM_ID)?,          false),
        ],
        data,
    })
}

pub fn derive_pump_fun_bonding_curve(mint: &Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[b"bonding-curve", mint.as_ref()],
        &Pubkey::from_str(PUMP_FUN_PROGRAM_ID).unwrap(),
    )
}

// ── Raydium CPMM ─────────────────────────────────────────────────────────────

#[derive(BorshSerialize, BorshDeserialize)]
pub struct RaydiumCpmmSwapArgs { pub amount_in: u64, pub minimum_amount_out: u64 }

pub struct CpmmPoolAccounts {
    pub pool_id: Pubkey,
    pub pool_authority: Pubkey,
    pub pool_state: Pubkey,
    pub token0_mint: Pubkey,
    pub token1_mint: Pubkey,
    pub token0_vault: Pubkey,
    pub token1_vault: Pubkey,
    pub lp_mint: Pubkey,
    pub observation_key: Pubkey,
}

pub fn build_raydium_cpmm_swap(
    pool: &CpmmPoolAccounts, user: &Pubkey,
    amount_in: u64, minimum_amount_out: u64, is_buy: bool,
) -> Result<Instruction> {
    let (input_mint, output_mint, input_vault, output_vault) = if is_buy {
        (pool.token0_mint, pool.token1_mint, pool.token0_vault, pool.token1_vault)
    } else {
        (pool.token1_mint, pool.token0_mint, pool.token1_vault, pool.token0_vault)
    };

    let user_in  = get_associated_token_address(user, &input_mint);
    let user_out = get_associated_token_address(user, &output_mint);

    let args = RaydiumCpmmSwapArgs { amount_in, minimum_amount_out };
    let mut data = vec![143u8, 190, 90, 218, 196, 30, 51, 222]; // swapBaseInput discriminator
    data.extend_from_slice(&borsh::to_vec(&args)?);

    Ok(Instruction {
        program_id: Pubkey::from_str(RAYDIUM_CPMM_PROGRAM_ID)?,
        accounts: vec![
            AccountMeta::new(        *user,                                         true),
            AccountMeta::new_readonly(Pubkey::from_str(RAYDIUM_CPMM_AUTHORITY)?,   false),
            AccountMeta::new_readonly(pool.pool_state,                              false),
            AccountMeta::new(        user_in,                                       false),
            AccountMeta::new(        user_out,                                      false),
            AccountMeta::new(        input_vault,                                   false),
            AccountMeta::new(        output_vault,                                  false),
            AccountMeta::new_readonly(Pubkey::from_str(TOKEN_PROGRAM_ID)?,          false),
            AccountMeta::new_readonly(Pubkey::from_str(TOKEN_2022_PROGRAM_ID)?,     false),
            AccountMeta::new_readonly(input_mint,                                   false),
            AccountMeta::new_readonly(output_mint,                                  false),
            AccountMeta::new(        pool.observation_key,                          false),
        ],
        data,
    })
}

// ── Raydium AMM v4 ────────────────────────────────────────────────────────────

#[derive(BorshSerialize, BorshDeserialize)]
pub struct RaydiumAmmSwapArgs {
    pub instruction: u8,
    pub amount_in: u64,
    pub minimum_amount_out: u64,
}

pub struct AmmV4PoolAccounts {
    pub amm_id: Pubkey,
    pub amm_authority: Pubkey,
    pub amm_open_orders: Pubkey,
    pub amm_target_orders: Pubkey,
    pub pool_coin_token_account: Pubkey,
    pub pool_pc_token_account: Pubkey,
    pub serum_program_id: Pubkey,
    pub serum_market: Pubkey,
    pub serum_bids: Pubkey,
    pub serum_asks: Pubkey,
    pub serum_event_queue: Pubkey,
    pub serum_coin_vault: Pubkey,
    pub serum_pc_vault: Pubkey,
    pub serum_vault_signer: Pubkey,
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
}

pub fn build_raydium_amm_swap(
    pool: &AmmV4PoolAccounts, user: &Pubkey,
    amount_in: u64, minimum_amount_out: u64, is_buy: bool,
) -> Result<Instruction> {
    let (source_mint, dest_mint) = if is_buy {
        (pool.quote_mint, pool.base_mint)
    } else {
        (pool.base_mint, pool.quote_mint)
    };
    let src  = get_associated_token_address(user, &source_mint);
    let dest = get_associated_token_address(user, &dest_mint);

    let args = RaydiumAmmSwapArgs { instruction: 9, amount_in, minimum_amount_out };
    let data = borsh::to_vec(&args)?;

    Ok(Instruction {
        program_id: Pubkey::from_str(RAYDIUM_AMM_PROGRAM_ID)?,
        accounts: vec![
            AccountMeta::new_readonly(Pubkey::from_str(TOKEN_PROGRAM_ID)?, false),
            AccountMeta::new(pool.amm_id,                   false),
            AccountMeta::new_readonly(pool.amm_authority,   false),
            AccountMeta::new(pool.amm_open_orders,          false),
            AccountMeta::new(pool.amm_target_orders,        false),
            AccountMeta::new(pool.pool_coin_token_account,  false),
            AccountMeta::new(pool.pool_pc_token_account,    false),
            AccountMeta::new_readonly(pool.serum_program_id, false),
            AccountMeta::new(pool.serum_market,             false),
            AccountMeta::new(pool.serum_bids,               false),
            AccountMeta::new(pool.serum_asks,               false),
            AccountMeta::new(pool.serum_event_queue,        false),
            AccountMeta::new(pool.serum_coin_vault,         false),
            AccountMeta::new(pool.serum_pc_vault,           false),
            AccountMeta::new_readonly(pool.serum_vault_signer, false),
            AccountMeta::new(src,    false),
            AccountMeta::new(dest,   false),
            AccountMeta::new(*user,  true),
        ],
        data,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Apply slippage tolerance: reduce minimum output by `slippage_bps` basis points.
pub fn apply_slippage(expected_out: u64, slippage_bps: u32) -> u64 {
    expected_out.saturating_sub((expected_out * slippage_bps as u64) / 10_000)
}

/// Estimate price impact for a CPMM constant-product pool.
pub fn estimate_price_impact_cpmm(reserve_in: u64, reserve_out: u64, amount_in: u64) -> f64 {
    if reserve_in == 0 || reserve_out == 0 { return 1.0; }
    let amount_out = (amount_in as u128 * reserve_out as u128)
        / (reserve_in as u128 + amount_in as u128);
    let spot = reserve_out as f64 / reserve_in as f64;
    let exec = amount_out as f64 / amount_in as f64;
    (spot - exec) / spot
}

/// Wrap native SOL into wSOL ATA, ready for token-program swaps.
pub fn wrap_sol_instructions(user: &Pubkey, lamports: u64) -> Result<Vec<Instruction>> {
    let wsol          = Pubkey::from_str(WSOL_MINT)?;
    let user_wsol_ata = get_associated_token_address(user, &wsol);
    let tok_prog      = Pubkey::from_str(TOKEN_PROGRAM_ID)?;

    let create = spl_associated_token_account::instruction::create_associated_token_account_idempotent(
        user, user, &wsol, &tok_prog,
    );
    let transfer = solana_sdk::system_instruction::transfer(user, &user_wsol_ata, lamports);
    // SyncNative instruction (discriminator 17)
    let sync = Instruction {
        program_id: tok_prog,
        accounts: vec![AccountMeta::new(user_wsol_ata, false)],
        data: vec![17u8],
    };
    Ok(vec![create, transfer, sync])
}

/// Close the wSOL ATA, converting its balance back to native SOL.
pub fn unwrap_wsol_instruction(user: &Pubkey) -> Result<Instruction> {
    let wsol          = Pubkey::from_str(WSOL_MINT)?;
    let user_wsol_ata = get_associated_token_address(user, &wsol);
    // CloseAccount instruction (discriminator 9)
    Ok(Instruction {
        program_id: Pubkey::from_str(TOKEN_PROGRAM_ID)?,
        accounts: vec![
            AccountMeta::new(        user_wsol_ata, false),
            AccountMeta::new(        *user,         false),
            AccountMeta::new_readonly(*user,        true),
        ],
        data: vec![9u8],
    })
}

/// Create an associated token account idempotently (does nothing if it already exists).
pub fn create_ata_idempotent(user: &Pubkey, mint: &Pubkey, token_2022: bool) -> Result<Instruction> {
    let prog = Pubkey::from_str(
        if token_2022 { TOKEN_2022_PROGRAM_ID } else { TOKEN_PROGRAM_ID },
    )?;
    Ok(spl_associated_token_account::instruction::create_associated_token_account_idempotent(
        user, user, mint, &prog,
    ))
}
