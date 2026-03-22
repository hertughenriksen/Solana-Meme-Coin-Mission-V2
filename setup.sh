#!/usr/bin/env bash
# setup.sh — One-command server setup for the Solana Memecoin Bot
# Run once on a fresh Ubuntu 22.04+ server or WSL2 Ubuntu instance.
# Usage: chmod +x setup.sh && ./setup.sh
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════╗"
echo -e "║     SOLANA MEMECOIN BOT — SETUP          ║"
echo -e "╚══════════════════════════════════════════╝${NC}"

OS=$(uname -s)

echo -e "\n${YELLOW}[1/8] Installing system dependencies...${NC}"
if [[ "$OS" == "Linux" ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential pkg-config libssl-dev git curl wget jq \
        postgresql postgresql-contrib redis-server \
        python3.11 python3.11-dev python3-pip python3-venv \
        protobuf-compiler libprotobuf-dev
    sudo systemctl enable --now postgresql redis-server
fi
echo -e "${GREEN}✓ System dependencies installed${NC}"

echo -e "\n${YELLOW}[2/8] Installing Rust toolchain...${NC}"
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi
rustup update stable
echo -e "${GREEN}✓ Rust $(rustc --version)${NC}"

echo -e "\n${YELLOW}[3/8] Setting up Python environment...${NC}"
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools --quiet

echo "  Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "  Installing torch-geometric..."
pip install torch-geometric --quiet
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --quiet 2>/dev/null || \
pip install torch-scatter torch-sparse --quiet 2>/dev/null || \
echo -e "${YELLOW}  ⚠ torch-scatter/sparse install failed — GNN training will still work via CPU${NC}"

echo "  Installing ML packages..."
pip install \
    catboost lightgbm xgboost scikit-learn imbalanced-learn optuna shap \
    transformers datasets tokenizers accelerate \
    onnx onnxruntime onnxmltools \
    --quiet

echo "  Installing data + DB packages..."
pip install \
    pandas numpy sqlalchemy psycopg2-binary asyncpg aiohttp \
    python-dotenv tqdm rich redis \
    --quiet

deactivate
echo -e "${GREEN}✓ Python environment ready at .venv/${NC}"

echo -e "\n${YELLOW}[4/8] Configuring PostgreSQL...${NC}"
sudo -u postgres psql -c "CREATE USER bot WITH PASSWORD 'password';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE memecoin_bot OWNER bot;" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE memecoin_bot TO bot;" 2>/dev/null || true
sudo -u postgres psql memecoin_bot -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;" 2>/dev/null || true
sudo -u postgres psql memecoin_bot -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;" 2>/dev/null || true

if [ -f "migrations/001_initial_schema.sql" ]; then
    PGPASSWORD=password psql -U bot -d memecoin_bot -h localhost \
        -f migrations/001_initial_schema.sql 2>/dev/null && \
        echo -e "${GREEN}✓ Database schema applied${NC}" || \
        echo -e "${YELLOW}⚠ Schema may already exist — continuing${NC}"
else
    echo -e "${YELLOW}⚠ migrations/001_initial_schema.sql not found — run manually after setup${NC}"
fi
echo -e "${GREEN}✓ PostgreSQL ready (db: memecoin_bot, user: bot)${NC}"

echo -e "\n${YELLOW}[5/8] Checking Redis...${NC}"
if redis-cli ping 2>/dev/null | grep -q PONG; then
    echo -e "${GREEN}✓ Redis running${NC}"
else
    sudo systemctl start redis-server
    sleep 1
    redis-cli ping | grep -q PONG && echo -e "${GREEN}✓ Redis started${NC}" || \
        echo -e "${RED}✗ Redis not responding — start manually: sudo systemctl start redis-server${NC}"
fi

echo -e "\n${YELLOW}[6/8] Setting up wallet keypairs...${NC}"
mkdir -p secrets && chmod 700 secrets

if ! command -v solana-keygen &>/dev/null; then
    echo "  Installing Solana CLI..."
    sh -c "$(curl -sSfL https://release.solana.com/stable/install)" 2>/dev/null
    export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
    SHELL_PROFILE="$HOME/.bashrc"
    if [[ "$SHELL" == *"zsh"* ]]; then SHELL_PROFILE="$HOME/.zshrc"; fi
    echo 'export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"' >> "$SHELL_PROFILE"
fi

if [ ! -f secrets/keypair.json ]; then
    solana-keygen new --no-passphrase --outfile secrets/keypair.json
    chmod 600 secrets/keypair.json
    PUBKEY=$(solana-keygen pubkey secrets/keypair.json 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ Trading keypair generated${NC}"
    echo -e "${RED}  ⚠  PUBLIC KEY: ${PUBKEY}${NC}"
    echo -e "${RED}  ⚠  BACK UP secrets/keypair.json BEFORE FUNDING THIS WALLET${NC}"
else
    echo -e "${GREEN}✓ Trading keypair already exists${NC}"
fi

if [ ! -f secrets/jito_keypair.json ]; then
    solana-keygen new --no-passphrase --outfile secrets/jito_keypair.json
    chmod 600 secrets/jito_keypair.json
    echo -e "${GREEN}✓ Jito auth keypair generated (no SOL needed in this wallet)${NC}"
else
    echo -e "${GREEN}✓ Jito keypair already exists${NC}"
fi

echo -e "\n${YELLOW}[7/8] Setting up environment file...${NC}"
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ .env created from .env.example — fill in your API keys${NC}"
    fi
else
    echo -e "${GREEN}✓ .env already exists${NC}"
fi

if [ ! -f config/local.toml ]; then
    if [ -f config/default.toml ]; then
        cp config/default.toml config/local.toml
        echo -e "${GREEN}✓ config/local.toml created from default.toml${NC}"
        echo -e "${YELLOW}  → Edit config/local.toml and replace all YOUR_* values${NC}"
    fi
fi

echo -e "\n${YELLOW}[8/8] Building Rust bot (5-15 minutes on first run)...${NC}"
source "$HOME/.cargo/env"

if cargo build --release 2>&1 | tail -5; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${YELLOW}⚠ Build failed — API keys may not be set yet.${NC}"
    echo -e "${YELLOW}  Fill in config/local.toml and run: cargo build --release${NC}"
fi

echo -e ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗"
echo -e "║  Setup complete! Next steps:                             ║"
echo -e "║                                                          ║"
echo -e "║  1. Fill in config/local.toml with your API keys        ║"
echo -e "║     (replace all YOUR_* placeholders)                   ║"
echo -e "║                                                          ║"
echo -e "║  2. Start services if not running:                      ║"
echo -e "║     sudo systemctl start postgresql redis-server        ║"
echo -e "║                                                          ║"
echo -e "║  3. Run the bot in dry mode (NO real trades):           ║"
echo -e "║     cargo run --release --bin bot                       ║"
echo -e "║                                                          ║"
echo -e "║  4. Dashboard → http://localhost:8080                   ║"
echo -e "║  5. Training → http://localhost:8080/training           ║"
echo -e "║                                                          ║"
echo -e "║  6. Run dry mode for 2+ weeks to collect training data  ║"
echo -e "║                                                          ║"
echo -e "║  7. Train ML models:                                    ║"
echo -e "║     source .venv/bin/activate                           ║"
echo -e "║     python ml/scripts/train_all.py                      ║"
echo -e "║                                                          ║"
echo -e "║  8. Only then set dry_run=false with tiny capital       ║"
echo -e "╚══════════════════════════════════════════════════════════╝${NC}"
echo -e ""
echo -e "${RED}⚠  NEVER trade funds you cannot afford to lose entirely."
echo -e "⚠  Run dry_run=true for at least 2 weeks first."
echo -e "⚠  Back up secrets/keypair.json and secrets/jito_keypair.json.${NC}"
