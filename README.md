# Solana Memecoin Bot

A high-frequency Solana memecoin trading bot with an ML-powered signal filter. Written in Rust with a Python ML training pipeline.

> ⚠️ **This software is provided as-is with no guarantees of profitability. Memecoins are extremely high-risk. Over 98% of Pump.fun tokens fail within 24 hours. Never trade funds you cannot afford to lose entirely. Run in dry-run mode for at least 2 weeks before enabling live trading.**

---

## Features

| Component | Details |
|-----------|---------|
| **Signal sources** | Yellowstone gRPC (sub-50ms on-chain), Twitter filtered stream, Telegram MTProto, copy-trading |
| **ML ensemble** | CatBoost + LightGBM tabular · Price Transformer · Wallet GNN · FinBERT sentiment |
| **Strategies** | New launch sniping · Copy-trading · Social sentiment momentum |
| **Execution** | Jito bundles submitted to 5 block engines in parallel for MEV protection |
| **Position management** | Split-entry (3 tranches) · Tiered take-profit · Trailing stop · Kelly criterion sizing |
| **Safety** | Circuit breaker · Daily loss cap · Reserve SOL floor · Dry-run mode |
| **Monitoring** | Live web dashboard (port 8080) · Prometheus metrics (port 9090) · Discord alerts |

---

## Architecture

```
Yellowstone gRPC ──┐
Twitter Stream     ──┤──► broadcast channel ──► StrategyEngine ──► ExecutionEngine
Telegram MTProto   ──┤         (signals)          (filter/ML)      (Jito bundles)
WalletTracker      ──┘
                                                       │
                                              PostgreSQL + Redis
                                               (trades, state)
                                                       │
                                               PriceFeed (2s poll)
                                               Dashboard (port 8080)
                                               Prometheus (port 9090)
```

### Signal flow

1. Signal sources emit `TokenSignal` structs onto a Tokio broadcast channel.
2. `StrategyEngine` receives each signal, checks circuit breaker and position limits, then runs the `FilterPipeline`.
3. `FilterPipeline` applies a 10-stage fast-fail filter (liquidity → market cap → token age → dev holdings → sniper concentration → mint authority → freeze authority → LP lock → creator history → price impact) then runs the 4-model ML ensemble in parallel.
4. Passing signals become `TradeDecision` structs sent to `ExecutionEngine`.
5. `ExecutionEngine` builds swap instructions, wraps them in a Jito bundle, and submits to all 5 block engines concurrently.
6. `PriceFeed` polls Jupiter every 2 seconds and writes prices to Redis.
7. Position manager ticks every 2 seconds, reading prices from Redis to evaluate TP/SL triggers.

---

## File Map

```
├── Cargo.toml
├── .env.example
├── .gitignore
├── setup.sh                       # One-command server setup
├── config/
│   ├── default.toml               # All parameters — safe to commit
│   └── local.toml                 # Your API keys — NEVER commit (git-ignored)
├── migrations/
│   ├── 001_initial_schema.sql     # Core tables: trades, tokens, signals, candles
│   ├── 002_add_source_wallet.sql  # Adds source_wallet to signals table
│   └── 003_add_training_sessions.sql
├── ml/
│   └── scripts/
│       ├── train_all.py           # Master training script (tabular + transformer)
│       ├── train_gnn.py           # Temporal Graph Attention Network
│       ├── train_nlp.py           # FinBERT fine-tuning for crypto sentiment
│       ├── gnn_sidecar.py         # Real-time GNN scoring service (Redis pub/sub)
│       └── outcome_tracker.py     # Labels token outcomes every 5 min for retraining
└── src/
    ├── main.rs                    # Entry point, spawns all async tasks
    ├── types.rs                   # All shared data types
    ├── config/mod.rs              # Config loader
    ├── db/
    │   ├── postgres.rs            # All SQL queries (sqlx)
    │   └── redis_client.rs        # Real-time state: positions, cache, circuit breaker
    ├── scanner/mod.rs             # Yellowstone gRPC scanner stub
    ├── signals/
    │   ├── twitter.rs             # X/Twitter v2 filtered stream
    │   ├── telegram.rs            # Telegram MTProto (grammers)
    │   └── wallet_tracker.rs      # Copy-trade wallet monitor
    ├── filter/mod.rs              # 10-stage filter pipeline
    ├── strategy/mod.rs            # 3 tracks: snipe / copy / sentiment
    ├── execution/
    │   ├── mod.rs                 # Wires everything, submits Jito bundles
    │   ├── swap_builder.rs        # Pump.fun + Raydium CPMM/AMM instructions
    │   └── rpc_client.rs          # Blockhash cache + multi-RPC failover
    ├── models/mod.rs              # ONNX inference: 4-model ensemble (heuristic until trained)
    ├── price_feed/mod.rs          # Jupiter price polling → Redis
    └── monitor/mod.rs             # Dashboard + Prometheus
```

---

## Requirements

### Server

| | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB SSD | 100 GB SSD |
| GPU | None | RTX 3090+ (for ML retraining only) |
| Location | Anywhere | Ashburn VA or Frankfurt (close to Jito engines) |

Windows users: use WSL2 (Ubuntu 22.04). Move WSL to a drive with at least 50 GB free.

### API Keys

| Service | Purpose | Required | Link |
|---------|---------|----------|------|
| **Helius** | Primary RPC + Yellowstone gRPC | ✅ Yes | https://helius.dev |
| **Birdeye** | Price data for ML labeling | ✅ Yes | https://birdeye.so |
| **Telegram** | MTProto group scanning | Recommended | https://my.telegram.org |
| **QuickNode** | Fallback RPC | Optional | https://quicknode.com |
| **Twitter/X** | Filtered stream API | Optional | https://developer.twitter.com |
| **bloXroute** | Parallel bundle relay | Optional | https://bloxroute.com |

Helius free tier is sufficient for basic RPC. Yellowstone gRPC (on-chain scanner) requires a **paid Helius plan**. The bot runs without it — Twitter and Telegram signals still work. Copy-trading also requires Yellowstone gRPC.

---

## Setup

### Option A — Automated (recommended)

```bash
git clone https://github.com/hertughenriksen/Solana-Meme-Coin-Mission-V2
cd Solana-Meme-Coin-Mission-V2
chmod +x setup.sh
./setup.sh
```

The script installs all system dependencies, Rust, Python, PostgreSQL, Redis, the Solana CLI, and builds the bot. It takes about 10–20 minutes on a fresh server.

After it completes, go to [Step 5 — Configure](#5-configure).

### Option B — Manual

#### 1. System dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev git curl \
    postgresql postgresql-contrib redis-server \
    python3.11 python3.11-dev python3-pip python3-venv protobuf-compiler

sudo systemctl enable --now postgresql redis-server
```

#### 2. Rust toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

#### 3. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# PyTorch (CUDA 12.1 — adjust for your GPU driver or use CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 2>/dev/null || \
pip install torch-scatter torch-sparse 2>/dev/null || true

# ML packages
pip install catboost lightgbm scikit-learn optuna transformers datasets \
    onnx onnxruntime onnxmltools \
    pandas numpy sqlalchemy psycopg2-binary asyncpg aiohttp \
    python-dotenv tqdm rich redis

deactivate
```

#### 4. Database

```bash
sudo -u postgres psql -c "CREATE USER bot WITH PASSWORD 'password';"
sudo -u postgres psql -c "CREATE DATABASE memecoin_bot OWNER bot;"
sudo -u postgres psql memecoin_bot -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"
sudo -u postgres psql memecoin_bot -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
PGPASSWORD=password psql -U bot -d memecoin_bot -h localhost \
    -f migrations/001_initial_schema.sql
```

#### 5. Wallet keypairs

The bot needs two keypairs: one for trading, one for Jito authentication.

```bash
mkdir -p secrets && chmod 700 secrets

# Install Solana CLI if not present
sh -c "$(curl -sSfL https://release.solana.com/stable/install)"
export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"

# Trading wallet
solana-keygen new --no-passphrase --outfile secrets/keypair.json
chmod 600 secrets/keypair.json

# Jito auth wallet (can be a fresh throwaway keypair — no SOL needed)
solana-keygen new --no-passphrase --outfile secrets/jito_keypair.json
chmod 600 secrets/jito_keypair.json
```

> ⚠️ **Back up both keypair files before funding the trading wallet. If you lose `secrets/keypair.json`, any funds in that wallet are gone forever.**

Print the public key of the trading wallet so you know where to send SOL:

```bash
solana-keygen pubkey secrets/keypair.json
```

### 5. Configure

```bash
cp config/default.toml config/local.toml
cp .env.example .env
```

Edit `config/local.toml` and replace all `YOUR_*` placeholders. The minimum required values are:

```toml
[rpc]
helius_api_key  = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
helius_rpc_url  = "https://mainnet.helius-rpc.com/?api-key=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
helius_ws_url   = "wss://mainnet.helius-rpc.com/?api-key=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

[signals.telegram]
api_id   = "12345678"
api_hash = "your_api_hash_here"
phone    = "+1234567890"
groups   = ["solana_alpha"]          # At least one group to monitor
```

Everything else has sensible defaults. See [Key Configuration Parameters](#key-configuration-parameters) for what each setting does.

### 6. Build

```bash
cargo build --release
```

First build takes 5–15 minutes. Subsequent builds are much faster due to caching.

### 7. Run in dry mode

```bash
sudo systemctl start postgresql redis-server
cargo run --release --bin bot
```

The bot starts with `dry_run = true` — it scans for signals and evaluates trades but **never sends real transactions**. Watch the dashboard at `http://localhost:8080` to confirm signals are being received and processed.

**Run dry mode for at least 2 weeks.** This collects the training data needed for the ML models.

---

## Training the ML Models

After 2+ weeks of dry-run data collection:

### Step 1 — Label outcomes

Start the outcome tracker. It runs every 5 minutes, fetching price data from Birdeye and labeling each token as a win (2× within 30 min) or loss:

```bash
source .venv/bin/activate
export BIRDEYE_API_KEY=your_key_here
export DATABASE_URL=postgresql://bot:password@localhost:5432/memecoin_bot
python ml/scripts/outcome_tracker.py &
```

Leave this running in a separate terminal (or set it up as a systemd service).

### Step 2 — Train tabular + transformer models

```bash
source .venv/bin/activate
python ml/scripts/train_all.py --db-url $DATABASE_URL
```

Trains CatBoost, LightGBM, and the Price Transformer. Exports ONNX files to `ml/models/`.

### Step 3 — Train the GNN

```bash
python ml/scripts/train_gnn.py --db-url $DATABASE_URL

# Or with synthetic data to test the pipeline first:
python ml/scripts/train_gnn.py --synthetic
```

### Step 4 — Fine-tune FinBERT sentiment

```bash
python ml/scripts/train_nlp.py
```

### Step 5 — Start the GNN sidecar

The GNN sidecar provides real-time wallet-graph scoring via Redis pub/sub. Start it and leave it running:

```bash
python ml/scripts/gnn_sidecar.py &
```

Models are saved to `ml/models/` and loaded automatically on the next bot restart.

---

## Going Live

Once models are trained and you have at least 2 weeks of dry-run data:

1. **Edit `config/local.toml`:**

```toml
[bot]
dry_run = false

[wallet]
max_position_size_sol = 0.05   # Start very small
max_total_capital_sol = 1.0    # Never risk more than this
```

2. **Fund the trading wallet** with the amount in `max_total_capital_sol` plus a small buffer for fees. Check the public key with:

```bash
solana-keygen pubkey secrets/keypair.json
```

3. **Start the bot:**

```bash
cargo run --release --bin bot
```

**Scaling advice:** Start at 0.05 SOL per trade. Double the ceiling only after 2 weeks at a sustained win rate above 55%. Never disable the circuit breaker.

---

## Key Configuration Parameters

All parameters live in `config/default.toml`. Override in `config/local.toml`. You never need to touch `default.toml` directly.

### Bot

```toml
[bot]
dry_run = true                     # ALWAYS start true; set false only after training
max_trades_per_hour = 60
trading_window_start_utc = 0       # 0 = midnight UTC (run all day)
trading_window_end_utc = 24
```

### Wallet / Capital

```toml
[wallet]
max_total_capital_sol = 10.0       # Hard cap — never risk more than this
max_position_size_sol = 0.3        # Per-trade ceiling
max_concurrent_positions = 8       # Open positions at once
reserve_sol = 1.0                  # Always kept for fees, never traded
```

### Filter

```toml
[filter]
min_liquidity_usd = 50000          # Minimum pool depth
max_market_cap_usd = 1000000       # Avoid tokens already pumped
min_win_probability = 0.55         # ML threshold — raise to 0.70+ after real training
require_mint_authority_disabled = true
require_freeze_authority_disabled = true
max_creator_rug_history = 0        # Zero tolerance for repeat ruggers
min_lp_lock_days = 30
```

### Strategy / Position Management

```toml
[strategy]
tp_1_multiplier = 1.5              # Sell 30% at +50%
tp_2_multiplier = 2.0              # Sell 30% at +100%
tp_3_trailing_stop_pct = 0.25      # Trail remaining 40% with 25% stop
hard_stop_loss_pct = 0.20          # Hard exit at -20%
time_stop_minutes = 60             # Exit if no +50% gain in 60 min
circuit_breaker_consecutive_losses = 3
circuit_breaker_pause_minutes = 30
```

### Jito MEV Protection

```toml
[jito]
tip_min_lamports = 10000           # 0.00001 SOL floor
tip_max_lamports = 5000000         # 0.005 SOL ceiling
tip_profit_share = 0.50            # Pay 50% of expected profit as tip
```

---

## Monitoring

| URL | Description |
|-----|-------------|
| `http://localhost:8080/` | Live trading dashboard (auto-refreshes every 5 s) |
| `http://localhost:8080/training` | Training progress dashboard |
| `http://localhost:8080/api/stats` | Session stats as JSON |
| `http://localhost:8080/api/positions` | Open positions as JSON |
| `http://localhost:8080/api/training-stats` | Training progress as JSON |
| `http://localhost:8080/health` | Health check (`ok`) |
| `http://localhost:9090/metrics` | Prometheus metrics |

Connect Grafana to the Prometheus endpoint (`http://localhost:9090/metrics`) for historical charting and alerting.

Discord alerts fire on trade entry, large losses, and circuit breaker trips. Set `alert_webhook_url` in `config/local.toml`.

---

## DEX Support

| DEX | Buy | Sell | Notes |
|-----|-----|------|-------|
| Pump.fun | ✅ | ✅ | Direct bonding-curve instruction |
| Raydium CPMM | ✅ | ✅ | `swapBaseInput` discriminator |
| Raydium AMM v4 | ✅ | ✅ | Requires Serum market accounts |
| Meteora / Orca / Others | ✅ | ✅ | Jupiter API fallback (~80 ms overhead) |

---

## Signal Sources

**Yellowstone gRPC** (paid Helius plan required) — watches every transaction on-chain in real time. Detects new Pump.fun launches and smart-wallet buys within ~50 ms of confirmation. Most reliable signal source.

**Telegram MTProto** — monitors configured alpha groups for contract address mentions. Scores message velocity and sentiment. Works with no paid API.

**Twitter/X Filtered Stream** — monitors KOL accounts and keyword triggers. Requires Twitter developer access (Basic tier or above).

**Copy Trading** — mirrors buys from a list of high-win-rate wallets. Requires Yellowstone gRPC to detect wallet transactions in real time.

---

## Weekly Retraining

Every Sunday at 02:00 UTC the bot triggers model retraining automatically. Run manually at any time:

```bash
source .venv/bin/activate
python ml/scripts/train_all.py --db-url $DATABASE_URL
python ml/scripts/train_gnn.py --db-url $DATABASE_URL
python ml/scripts/train_nlp.py
```

---

## Troubleshooting

**Build fails on first run**
Ensure you are using the exact dependency versions in `Cargo.toml`. The `solana-sdk = "2.0"` and `sqlx = "0.8"` pinned versions are required to avoid `zeroize` conflicts.

**PostgreSQL connection refused**
```bash
sudo systemctl start postgresql
sudo systemctl status postgresql
```

**Redis connection refused**
```bash
sudo systemctl start redis-server
redis-cli ping   # should print PONG
```

**Telegram auth loop**
Delete `secrets/telegram.session` and restart. You will be prompted for your verification code again.

**Models returning 0.5 for everything**
Expected before training. Run `python ml/scripts/train_all.py` after collecting 2+ weeks of dry-run data.

**Bundle rejected by all Jito engines**
The bot retries with increasing slippage up to `slippage_max_bps`. If all retries fail during high congestion, raise `tip_min_lamports` in `config/local.toml`.

**Yellowstone / wallet tracker not working**
Both require a paid Helius gRPC endpoint. Set `yellowstone_grpc_url` and `yellowstone_grpc_token` in `config/local.toml`. Twitter and Telegram signals work without it.

**`cargo build` fails with `spl-associated-token-account` version error**
Make sure `Cargo.toml` specifies `spl-associated-token-account = { version = "4.0", ... }`. Version 6.0 does not exist on crates.io.

---

## Bugs Fixed vs. Original Code

The following bugs were identified and corrected from the initial codebase:

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `Cargo.toml` | `spl-associated-token-account = "6.0"` does not exist on crates.io | Changed to `"4.0"` |
| 2 | `Cargo.toml` / `redis_client.rs` | `AsyncCommands::incr_by_float` does not exist on `redis = "0.24"` — compile error | Bumped redis to `"0.25"`, replaced calls with raw `INCRBYFLOAT` command |
| 3 | `monitor/mod.rs` | `api_training_start` handler created a `sqlx::query!` value then dropped it without calling `.execute()` — training sessions were never recorded | Added `Database::start_training_session()` and called it from the handler |
| 4 | `monitor/mod.rs` | Training dashboard HTML had `{btn_disabled}` as a literal string inside a CSS class — it rendered in the browser as the text `{btn_disabled}` | Built button attributes as Rust `let` bindings before the `format!` call |
| 5 | `monitor/mod.rs` | Live dashboard used positional `{}` format placeholders in a complex template leading to argument order mismatch | Rewrote template to use named format arguments |
| 6 | `ml/scripts/train_all.py` | `onnxmltools.convert_catboost` was removed in recent onnxmltools releases — import error at training time | Replaced with CatBoost native ONNX export: `model.save_model(path, format="onnx")` |

---

## Risk Warnings

- Memecoins are extremely high risk. Over 98% of Pump.fun tokens fail within 24 hours.
- You are competing with professional HFT teams running co-located servers near validators.
- Never trade funds you cannot afford to lose entirely.
- Run dry-run for at least 2 weeks before enabling live trading.
- The circuit breaker pauses trading after 3 consecutive losses. Do not disable it.
- Keep `reserve_sol` funded at all times — transactions fail without SOL for fees.
- Back up `secrets/keypair.json` and `secrets/jito_keypair.json` before funding the wallet.
- This software is provided as-is with no guarantee of profitability.

---

## License

MIT
