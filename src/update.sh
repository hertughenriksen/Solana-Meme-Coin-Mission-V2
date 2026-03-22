#!/usr/bin/env bash
# update.sh — downloads the latest repo and applies all bug fixes + autoresearch
# Run from anywhere: bash update.sh

set -e  # stop on any error

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[•]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
error()   { echo -e "${RED}[✗]${NC} $1"; exit 1; }
header()  { echo -e "\n${BOLD}$1${NC}"; echo "────────────────────────────────────────"; }

# ── Repo config ───────────────────────────────────────────────────────────────
REPO_URL="https://github.com/hertughenriksen/Solana-Meme-Coin-Mission-V2"
PROJECT_DIR="/mnt/d/Solana-Meme-Coin-Mission-V2"

header "🤖 Memecoin Bot — Update Script"

# ── 0. Download / update the repo ─────────────────────────────────────────────
header "Step 0/6 — Downloading latest code from GitHub"

if ! command -v git &>/dev/null; then
    error "git is not installed.\n  Ubuntu/WSL: sudo apt install git\n  Windows:    https://git-scm.com/download/win"
fi

if [ -d "$PROJECT_DIR/.git" ]; then
    info "Repo already exists at $PROJECT_DIR — pulling latest changes..."
    cd "$PROJECT_DIR"
    git pull origin main 2>&1 || git pull origin master 2>&1 || warn "git pull failed — continuing with existing files"
    success "Repo updated"
else
    info "Cloning repo into $PROJECT_DIR..."
    mkdir -p "$(dirname "$PROJECT_DIR")"
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    success "Repo cloned"
fi

cd "$PROJECT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/src" ] && [ "$SCRIPT_DIR" != "$PROJECT_DIR" ]; then
    info "Copying patched files from zip into $PROJECT_DIR..."
    cp -r "$SCRIPT_DIR/src"        "$PROJECT_DIR/"
    cp -r "$SCRIPT_DIR/ml"         "$PROJECT_DIR/"
    cp -r "$SCRIPT_DIR/migrations" "$PROJECT_DIR/"
    cp    "$SCRIPT_DIR/Cargo.toml" "$PROJECT_DIR/"
    success "Patched files copied"
fi

if [ ! -f "Cargo.toml" ]; then
    error "Cargo.toml not found in $PROJECT_DIR — something went wrong with the clone"
fi

# ── 1. Load .env ──────────────────────────────────────────────────────────────
header "Step 1/6 — Loading environment"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    success ".env loaded"
else
    warn "No .env file found — will try to use DATABASE_URL from environment"
fi

if [ -z "$DATABASE_URL" ]; then
    error "DATABASE_URL is not set. Add it to your .env file:\n  DATABASE_URL=postgres://user:password@localhost:5432/dbname"
fi

success "DATABASE_URL found"

# ── 2. Check required tools ───────────────────────────────────────────────────
header "Step 2/6 — Checking tools"

check_tool() {
    if command -v "$1" &>/dev/null; then
        success "$1 found"
    else
        error "$1 is not installed. $2"
    fi
}

check_tool cargo  "Install Rust: https://rustup.rs"
check_tool python3 "Install Python 3.11+"

if command -v psql &>/dev/null; then
    success "psql found"
else
    warn "psql not found — attempting to install..."
    if command -v apt &>/dev/null; then
        sudo apt-get install -y postgresql-client
    elif command -v brew &>/dev/null; then
        brew install postgresql
    else
        error "Cannot install psql automatically. Install it manually:\n  Ubuntu: sudo apt install postgresql-client\n  Mac:    brew install postgresql"
    fi
    success "psql installed"
fi

# ── 3. Stop the bot if running ────────────────────────────────────────────────
header "Step 3/6 — Stopping bot"

BOT_STOPPED=false

if systemctl is-active --quiet memecoin-bot 2>/dev/null; then
    info "Stopping systemd service memecoin-bot..."
    sudo systemctl stop memecoin-bot
    BOT_STOPPED=true
    success "Service stopped"
elif command -v pm2 &>/dev/null && pm2 list 2>/dev/null | grep -q "bot"; then
    info "Stopping pm2 process..."
    pm2 stop bot 2>/dev/null || true
    BOT_STOPPED=true
    success "pm2 process stopped"
elif pgrep -x "bot" &>/dev/null; then
    info "Killing bot process..."
    pkill -x "bot" || true
    sleep 1
    BOT_STOPPED=true
    success "Process killed"
else
    warn "Bot doesn't appear to be running — continuing anyway"
fi

# ── 4. Run database migration ─────────────────────────────────────────────────
header "Step 4/6 — Database migration"

MIGRATION="migrations/004_autoresearch_columns.sql"

if [ ! -f "$MIGRATION" ]; then
    error "Migration file not found at $MIGRATION — make sure you extracted the zip into your project root"
fi

info "Running $MIGRATION..."
if psql "$DATABASE_URL" -f "$MIGRATION" 2>&1; then
    success "Migration applied"
else
    OUTPUT=$(psql "$DATABASE_URL" -f "$MIGRATION" 2>&1 || true)
    if echo "$OUTPUT" | grep -qi "ERROR" && ! echo "$OUTPUT" | grep -qi "already exists"; then
        echo "$OUTPUT"
        error "Migration failed — see error above"
    else
        warn "Migration had warnings (columns may already exist) — this is fine"
    fi
fi

# ── 5. Install Python dependencies ────────────────────────────────────────────
header "Step 5/6 — Python dependencies"

REQUIREMENTS="ml/autoresearch/requirements.txt"

if [ ! -f "$REQUIREMENTS" ]; then
    error "Requirements file not found at $REQUIREMENTS"
fi

if [ -f ".venv/bin/pip" ]; then
    PIP=".venv/bin/pip"
    info "Using existing virtualenv at .venv"
elif [ -f "venv/bin/pip" ]; then
    PIP="venv/bin/pip"
    info "Using existing virtualenv at venv"
elif command -v uv &>/dev/null; then
    info "uv found — using for faster install"
    uv pip install -r "$REQUIREMENTS"
    success "Python dependencies installed"
    PIP=""
else
    PIP="pip3"
    warn "No virtualenv found — installing to system Python (consider using a venv)"
fi

if [ -n "$PIP" ]; then
    info "Installing from $REQUIREMENTS (this may take a minute)..."
    $PIP install -r "$REQUIREMENTS" --quiet
    success "Python dependencies installed"
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    warn "ANTHROPIC_API_KEY not set in .env — autoresearch agent won't work without it"
    warn "Add this to your .env: ANTHROPIC_API_KEY=sk-ant-..."
else
    success "ANTHROPIC_API_KEY found"
fi

# ── 6. Rebuild Rust bot ───────────────────────────────────────────────────────
header "Step 6/6 — Rebuilding Rust bot"

info "Running cargo build --release (this takes a few minutes on first run)..."
if cargo build --release 2>&1; then
    success "Build complete"
else
    error "cargo build failed — check the errors above"
fi

# ── Done — restart ─────────────────────────────────────────────────────────────
header "✅ Update complete"

if [ "$BOT_STOPPED" = true ]; then
    echo ""
    info "Restarting bot..."
    if systemctl is-enabled --quiet memecoin-bot 2>/dev/null; then
        sudo systemctl start memecoin-bot
        success "Systemd service restarted"
    elif command -v pm2 &>/dev/null; then
        pm2 start bot 2>/dev/null || true
        success "pm2 process restarted"
    else
        warn "Restart your bot manually: ./target/release/bot"
    fi
fi

echo ""
echo -e "${GREEN}${BOLD}All done! Summary of what was applied:${NC}"
echo "  • Database migration 004 (autoresearch columns)"
echo "  • Python deps for autoresearch dashboard"
echo "  • Rust bot rebuilt with all bug fixes + ONNX model loading"
echo ""
echo -e "${BOLD}To start the autoresearch dashboard:${NC}"
echo "  uvicorn ml.autoresearch.dashboard:app --host 0.0.0.0 --port 8081"
echo ""
echo -e "${BOLD}To run the agent overnight (50 experiments, 8 hours):${NC}"
echo "  python -m ml.autoresearch.agent --db-url \"\$DATABASE_URL\" --max-experiments 50 --hours 8"
