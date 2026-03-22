# Solana Memecoin Bot

High-frequency Solana memecoin trading bot with ML-powered signal filtering.

- **Yellowstone gRPC** — sub-50ms on-chain signal detection
- **4-model ML ensemble** — Tabular (CatBoost + LightGBM) + Price Transformer + GNN + FinBERT sentiment
- **3 concurrent strategies** — new launch sniping, copy-trading, sentiment momentum
- **Jito bundles** — submitted to 5 block engines in parallel for MEV protection
- **Split-entry + trailing stop** — position management with Kelly criterion sizing
- **Live dashboard** — auto-refreshing web UI + Prometheus metrics

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
```

---

## File Map

```
├── Cargo.toml
├── .env.example
├── .gitignore
├── config/
│   ├── default.toml              # All parameters — commit this
│   └── local.toml                # Your API keys — NEVER commit this
├── migrations/
│   └── 001_initial_schema.sql    # PostgreSQL schema
├── ml/
│   └── scripts/
│       ├── train_all.py          # Master training script (tabular + transformer)
│       ├── train_gnn.py          # Temporal Graph Attention Network
│       ├── train_nlp.py          # FinBERT fine-tuning for crypto sentiment
│       ├── gnn_sidecar.py        # Real-time GNN scoring service (Redis pub/sub)
│       └── outcome_tracker.py    # Labels token outcomes every 5 min for retraining
└── src/
    ├── main.rs                   # Entry point, spawns all async tasks
    ├── types.rs                  # All shared data types
    ├── config/mod.rs             # Config loader (default.toml + local.toml)
    ├── db/
    │   ├── mod.rs
    │   ├── postgres.rs           # All SQL queries (sqlx)
    │   └── redis_client.rs       # Real-time state: positions, cache, circuit breaker
    ├── scanner/
    │   ├── mod.rs                # Yellowstone gRPC scanner
    │   └── instruction_parser.rs # Decodes raw tx bytes → Pump.fun/Raydium events
    ├── signals/
    │   ├── mod.rs
    │   ├── twitter.rs            # X/Twitter v2 filtered stream
    │   ├── telegram.rs           # Telegram MTProto (grammers)
    │   └── wallet_tracker.rs     # Copy-trade wallet monitor
    ├── filter/mod.rs             # 10-stage filter pipeline (fast → expensive)
    ├── strategy/mod.rs           # 3 tracks: snipe / copy / sentiment
    ├── execution/
    │   ├── mod.rs                # Wires everything, submits Jito bundles
    │   ├── swap_builder.rs       # Pump.fun + Raydium CPMM/AMM instructions
    │   └── rpc_client.rs         # Blockhash cache + multi-RPC failover
    ├── models/mod.rs             # ONNX inference: 4-model ensemble
    └── monitor/mod.rs            # Dashboard (port 8080) + Prometheus (port 9090)
```

---

## Setup

### 1. Server Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| RAM | 16GB | 32GB |
| Disk | 50GB SSD | 100GB SSD |
| GPU | None | RTX 3090+ (for retraining) |
| Location | Anywhere | Ashburn VA or Frankfurt (close to Jito) |

Windows users: use WSL2 (Ubuntu 22.04). Move WSL to a drive with at least 50GB free.

### 2. Install Dependencies

```bash
# System packages
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev git curl \
    postgresql postgresql-contrib redis-server \
    python3.11 python3.11-dev python3-pip python3-venv protobuf-compiler

sudo systemctl enable --now postgresql redis-server

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Python venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# ML dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install catboost lightgbm scikit-learn optuna transformers datasets \
    onnx onnxruntime onnxmltools pandas numpy sqlalchemy \
    psycopg2-binary asyncpg aiohttp python-dotenv
```

### 3. Database Setup

```bash
sudo -u postgres psql -c "CREATE USER bot WITH PASSWORD 'password';"
sudo -u postgres psql -c "CREATE DATABASE memecoin_bot OWNER bot;"
sudo -u postgres psql memecoin_bot -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"
PGPASSWORD=password psql -U bot -d memecoin_bot -h localhost -f migrations/001_initial_schema.sql
```

### 4. Generate Wallet Keypair

```bash
mkdir -p secrets && chmod 700 secrets

# Install Solana CLI if not present
sh -c "$(curl -sSfL https://release.solana.com/stable/install)"
export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"

solana-keygen new --no-passphrase --outfile secrets/keypair.json
chmod 600 secrets/keypair.json
```

> ⚠️ **Back up `secrets/keypair.json` before funding the wallet. If you lose it, funds are gone forever.**

### 5. API Keys

| Service | Purpose | Required | URL |
|---------|---------|----------|-----|
| **Helius** | Primary RPC + Yellowstone gRPC | ✅ Yes | https://helius.dev |
| **Birdeye** | Price data for ML labeling | ✅ Yes | https://birdeye.so |
| **Telegram** | MTProto group scanning | Recommended | https://my.telegram.org |
| **QuickNode** | Fallback RPC | Optional | https://quicknode.com |
| **Twitter/X** | Filtered stream API | Optional | https://developer.twitter.com |
| **bloXroute** | Parallel bundle relay | Optional | https://bloxroute.com |

Helius free tier is sufficient for basic RPC. Yellowstone gRPC (on-chain scanner) requires a paid Helius plan. The bot runs without it — Twitter and Telegram signals still work.

### 6. Configure

```bash
cp config/default.toml config/local.toml
cp .env.example .env
```

Edit `config/local.toml` and replace all `YOUR_*` placeholders with your real API keys. The most important ones:

```toml
[rpc]
helius_api_key = "your_key_here"
helius_rpc_url = "https://mainnet.helius-rpc.com/?api-key=your_key_here"

[signals.telegram]
api_id   = "12345678"
api_hash = "your_hash_here"
phone    = "+1234567890"
groups   = ["group_username_1", "group_username_2"]
```

### 7. Build

```bash
cargo build --release
```

First build takes 5–15 minutes. Subsequent builds are much faster.

### 8. Run in Dry Mode

```bash
# Make sure PostgreSQL and Redis are running
sudo systemctl start postgresql redis-server

cargo run --release --bin bot
```

The bot starts with `dry_run = true` — it scans for signals and evaluates trades but never sends real transactions. Watch the dashboard at `http://localhost:8080` to see it working.

**Run dry mode for at least 2 weeks before going live.** This collects the training data needed for the ML models.

---

## Training the ML Models

After 2+ weeks of dry-run data collection, label the tokens then train all models.

### Step 1: Start the Outcome Tracker

Runs every 5 minutes, labels tokens as wins (2x within 30 min) or losses:

```bash
source .venv/bin/activate
export BIRDEYE_API_KEY=your_key_here
export DATABASE_URL=postgresql://bot:password@localhost:5432/memecoin_bot
python ml/scripts/outcome_tracker.py
```

Leave this running in a separate terminal or set it up as a systemd service.

### Step 2: Train All Models

```bash
source .venv/bin/activate
python ml/scripts/train_all.py --db-url $DATABASE_URL
```

Trains CatBoost, LightGBM, and the Price Transformer. Exports ONNX files to `ml/models/`.

### Step 3: Train the GNN

```bash
python ml/scripts/train_gnn.py --db-url $DATABASE_URL
```

For testing without real data:

```bash
python ml/scripts/train_gnn.py --synthetic
```

### Step 4: Fine-tune FinBERT Sentiment

```bash
python ml/scripts/train_nlp.py
```

### Step 5: Start the GNN Sidecar

Provides real-time GNN scoring via Redis pub/sub:

```bash
python ml/scripts/gnn_sidecar.py &
```

Models are exported to `ml/models/` and loaded automatically on next bot restart.

---

## Going Live

Once models are trained and you have 2+ weeks of dry-run data:

```toml
# In config/local.toml
[bot]
dry_run = false

[wallet]
max_position_size_sol = 0.05   # Start very small
max_total_capital_sol = 1.0
```

Start with `0.05 SOL` per trade. Double the ceiling every 2 weeks only if win rate stays above 55%.

```bash
cargo run --release --bin bot
```

---

## Key Configuration Parameters

All parameters live in `config/default.toml`. Override in `config/local.toml`.

### Bot

```toml
[bot]
dry_run = true                    # ALWAYS start true
max_trades_per_hour = 60
trading_window_start_utc = 0      # 0 = midnight UTC
trading_window_end_utc = 24       # 24 = run all day
```

### Wallet / Capital

```toml
[wallet]
max_total_capital_sol = 10.0      # Hard cap — never risk more than this
max_position_size_sol = 0.3       # Per-trade limit
max_concurrent_positions = 8
reserve_sol = 1.0                 # Always keep this for fees
```

### Filter

```toml
[filter]
min_liquidity_usd = 50000
max_market_cap_usd = 1000000
min_win_probability = 0.55        # ML threshold — raise to 0.80 after real training
require_mint_authority_disabled = true
require_freeze_authority_disabled = true
max_creator_rug_history = 0       # Zero tolerance for serial ruggers
min_lp_lock_days = 30
```

### Strategy / Position Management

```toml
[strategy]
tp_1_multiplier = 1.5             # Sell 30% at +50%
tp_2_multiplier = 2.0             # Sell 30% at +100%
tp_3_trailing_stop_pct = 0.25     # Trail remaining 40% with 25% stop
hard_stop_loss_pct = 0.20         # Hard exit at -20%
time_stop_minutes = 60            # Exit if no 50% gain in 60 min
circuit_breaker_consecutive_losses = 3
circuit_breaker_pause_minutes = 30
```

### Jito Tips

```toml
[jito]
tip_min_lamports = 10000          # 0.00001 SOL floor
tip_max_lamports = 5000000        # 0.005 SOL ceiling
tip_profit_share = 0.50           # Pay 50% of estimated profit as tip
```

---

## Monitoring

| Endpoint | Description |
|---------|-------------|
| `http://localhost:8080/` | Live dashboard (auto-refreshes every 5s) |
| `http://localhost:8080/api/stats` | JSON session stats |
| `http://localhost:8080/api/positions` | Open positions JSON |
| `http://localhost:8080/health` | Health check |
| `http://localhost:9090/metrics` | Prometheus metrics |

Connect Grafana to the Prometheus endpoint for historical charting.

---

## Weekly Retraining

Every Sunday at 02:00 UTC the bot automatically triggers retraining. You can also run it manually at any time:

```bash
source .venv/bin/activate
python ml/scripts/train_all.py --db-url $DATABASE_URL
python ml/scripts/train_gnn.py --db-url $DATABASE_URL
python ml/scripts/train_nlp.py
```

---

## DEX Support

| DEX | Buy | Sell | Notes |
|-----|-----|------|-------|
| Pump.fun | ✅ | ✅ | Bonding curve — direct instruction |
| Raydium CPMM | ✅ | ✅ | swapBaseInput discriminator |
| Raydium AMM v4 | ✅ | ✅ | Requires Serum market accounts |
| Meteora | ✅ | ✅ | Routed via Jupiter API (~80ms overhead) |
| Others | ✅ | ✅ | Jupiter catch-all fallback |

---

## Signal Sources

**Yellowstone gRPC** (requires paid Helius) — watches every transaction on-chain in real time. Detects new token launches and smart wallet buys within ~50ms of confirmation. Most reliable signal source.

**Telegram MTProto** — monitors configured alpha groups for contract address mentions. Scores message velocity and sentiment. Works without any paid API.

**Twitter/X Filtered Stream** — monitors KOL accounts and keyword triggers. Requires Twitter developer access.

**Copy Trading** — mirrors buys from a configured list of high-win-rate wallets. Also requires Yellowstone gRPC to detect wallet transactions in real time.

---

## Troubleshooting

**Build fails with dependency errors** — make sure you are using the exact versions in `Cargo.toml`. The `solana-sdk = "2.0"` and `sqlx = "0.8"` versions are required to avoid `zeroize` conflicts with older versions.

**PostgreSQL connection refused**
```bash
sudo systemctl start postgresql
sudo systemctl status postgresql
```

**Redis connection refused**
```bash
sudo systemctl start redis-server
redis-cli ping   # should return PONG
```

**Telegram auth loop** — delete `secrets/telegram.session` and restart. You will be prompted for your verification code again.

**Models returning 0.5 for everything** — expected before training. Run `python ml/scripts/train_all.py` after collecting 2+ weeks of dry-run data.

**Bundle rejected by all Jito engines** — the bot automatically retries with higher slippage up to `slippage_max_bps`. If all retries fail during high congestion, raise `tip_min_lamports` in `config/local.toml`.

---

## Risk Warnings

- Memecoins are extremely high risk. Over 98% of Pump.fun tokens fail within 24 hours.
- You are competing with professional HFT teams running co-located servers near validators.
- Never trade funds you cannot afford to lose entirely.
- Run dry-run for at least 2 weeks before enabling live trading.
- The circuit breaker pauses trading after 3 consecutive losses. Do not disable it.
- This software is provided as-is with no guarantees of profitability.
