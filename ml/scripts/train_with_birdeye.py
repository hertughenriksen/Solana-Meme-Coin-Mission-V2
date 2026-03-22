"""
train_with_birdeye.py — Automatic Training with Birdeye API Data
Fetches top memecoin data from Birdeye, trains all ML models automatically.
Usage: python ml/scripts/train_with_birdeye.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# Get API key from environment
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "cc26595c7d3f433e9604c2ef7c1b4cda")
BIRDEYE_BASE_URL = "https://public-api.birdeye.so"

# Rate limiting: 60 requests per minute = 1 request per second
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_SECONDS = 60
MIN_REQUEST_INTERVAL = RATE_LIMIT_SECONDS / RATE_LIMIT_REQUESTS  # 1 second between requests

# Track last request time
last_request_time = 0

def rate_limited_request(url, **kwargs):
    """Make rate-limited API request"""
    global last_request_time
    
    # Calculate time since last request
    current_time = time.time()
    time_since_last = current_time - last_request_time
    
    # Sleep if necessary to respect rate limit
    if time_since_last < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - time_since_last
        log.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
    
    # Update last request time
    last_request_time = time.time()
    
    # Make the request
    return requests.get(url, **kwargs)

def fetch_token_list(limit=100):
    """Fetch top tokens from Birdeye API"""
    log.info(f"Fetching top {limit} tokens from Birdeye API...")
    
    url = f"{BIRDEYE_BASE_URL}/v1/token/token_list"
    params = {
        "sort_by": "holder",
        "sort_type": "desc",
        "limit": min(limit, 100),
        "offset": 0
    }
    headers = {"X-API-Key": BIRDEYE_API_KEY}
    
    try:
        resp = rate_limited_request(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "data" in data and "tokens" in data["data"]:
            tokens = data["data"]["tokens"]
            log.info(f"✓ Fetched {len(tokens)} tokens")
            return tokens
        else:
            log.warning(f"Unexpected response format: {data}")
            return []
    except Exception as e:
        log.error(f"Failed to fetch tokens: {e}")
        return []

def fetch_token_info(mint):
    """Fetch detailed info for a single token"""
    url = f"{BIRDEYE_BASE_URL}/v1/token/meta"
    params = {"address": mint}
    headers = {"X-API-Key": BIRDEYE_API_KEY}
    
    try:
        resp = rate_limited_request(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", {})
    except Exception as e:
        log.warning(f"Failed to fetch info for {mint}: {e}")
        return {}

def fetch_token_price_history(mint, days=7):
    """Fetch price history for a token"""
    url = f"{BIRDEYE_BASE_URL}/v1/token/price_history"
    params = {
        "address": mint,
        "type": "1h",
        "time_from": int((datetime.now() - timedelta(days=days)).timestamp()),
        "time_to": int(datetime.now().timestamp())
    }
    headers = {"X-API-Key": BIRDEYE_API_KEY}
    
    try:
        resp = rate_limited_request(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("data", {}).get("items", [])
        return items
    except Exception as e:
        log.warning(f"Failed to fetch price history for {mint}: {e}")
        return []

def build_training_data(tokens, limit_per_token=50):
    """Build training dataset from Birdeye token data"""
    log.info(f"Building training data from {len(tokens)} tokens...")
    
    rows = []
    
    for i, token in enumerate(tokens[:20]):  # Limit to top 20 for speed
        mint = token.get("address", "")
        if not mint:
            continue
        
        log.info(f"[{i+1}/20] Processing {token.get('name', 'Unknown')} ({mint[:8]}...)")
        
        # Get token info
        info = fetch_token_info(mint)
        
        # Get price history
        history = fetch_token_price_history(mint, days=7)
        
        if not history or len(history) < 5:
            log.warning(f"  Not enough price data for {mint}")
            continue
        
        # Extract features from price history
        prices = [h.get("value", 0) for h in history]
        volumes = [h.get("v", 0) for h in history]
        
        if not prices or prices[-1] == 0:
            continue
        
        # Calculate features
        price_change_1h = ((prices[-1] - prices[0]) / prices[0] * 100) if prices[0] > 0 else 0
        avg_volume = np.mean(volumes) if volumes else 0
        volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        row = {
            "mint": mint,
            "name": token.get("name", ""),
            "symbol": token.get("symbol", ""),
            "liquidity_usd": float(info.get("liquidity", 0)),
            "market_cap_usd": float(info.get("mc", 0)),
            "volume_usd_1h": avg_volume,
            "price_change_5m_pct": price_change_1h,
            "price_change_1h_pct": price_change_1h,
            "holder_count": float(info.get("holder", 0)),
            "token_age_seconds": int((datetime.now() - datetime.fromisoformat(
                info.get("createdAt", datetime.now().isoformat()).replace('Z', '+00:00')
            )).total_seconds()) if "createdAt" in info else 3600,
            "volatility": volatility,
            "price_current": float(prices[-1]) if prices else 0,
            "first_seen_at": datetime.now(),
        }
        
        # Generate synthetic label (positive if good metrics)
        score = 0
        if row["liquidity_usd"] > 50000:
            score += 1
        if row["market_cap_usd"] > 100000:
            score += 1
        if row["holder_count"] > 100:
            score += 1
        if 0 < row["volatility"] < 0.5:
            score += 1
        
        row["ml_label"] = 1 if score >= 2 else 0
        rows.append(row)
    
    df = pd.DataFrame(rows)
    log.info(f"✓ Built dataset with {len(df)} tokens | Positive rate: {df['ml_label'].mean():.1%}")
    return df

def save_training_data(df, output_path="ml/models/training_data.parquet"):
    """Save training data to parquet"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    log.info(f"✓ Saved training data to {output_path}")
    return output_path

def trigger_model_training():
    """Call the main training pipeline"""
    log.info("Triggering main training pipeline...")
    
    try:
        # Change to ml/scripts directory
        os.chdir(Path(__file__).parent)
        
        # Import and run train_all
        import subprocess
        result = subprocess.run(
            [sys.executable, "train_all.py", "--data-dir", "../models"],
            capture_output=False
        )
        
        if result.returncode == 0:
            log.info("✓ Model training completed successfully")
            return True
        else:
            log.error(f"Model training failed with code {result.returncode}")
            return False
    except Exception as e:
        log.error(f"Failed to trigger training: {e}")
        return False

def main():
    log.info("=" * 60)
    log.info("BIRDEYE-POWERED AUTOMATIC TRAINING PIPELINE")
    log.info("=" * 60)
    
    # Fetch token list
    tokens = fetch_token_list(limit=100)
    if not tokens:
        log.error("Failed to fetch tokens. Check Birdeye API key.")
        return False
    
    # Build training data
    df = build_training_data(tokens)
    if df.empty:
        log.error("Failed to build training dataset")
        return False
    
    # Save data
    data_path = save_training_data(df)
    
    # Trigger training
    log.info("=" * 60)
    log.info("PHASE 2: TRAINING MODELS")
    log.info("=" * 60)
    
    success = trigger_model_training()
    
    if success:
        log.info("=" * 60)
        log.info("✅ TRAINING COMPLETE")
        log.info("Models saved to: ml/models/")
        log.info("=" * 60)
        return True
    else:
        log.warning("Training encountered issues, but data was collected")
        return False

if __name__ == "__main__":
    main()
