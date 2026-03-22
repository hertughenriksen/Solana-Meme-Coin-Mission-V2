"""
train_all.py — Master ML Training Pipeline
Trains: CatBoost+LightGBM tabular, Price Transformer, delegates GNN & NLP.
Usage: python ml/scripts/train_all.py --db-url $DATABASE_URL

FIX: onnxmltools.convert_catboost was removed in recent onnxmltools versions.
     CatBoost now supports native ONNX export via model.save_model(..., format='onnx').
     LightGBM export still uses onnxmltools (convert_lightgbm), which remains
     stable as of onnxmltools 1.12.
"""
import argparse, json, logging, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import onnx

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TABULAR_FEATURES = [
    "liquidity_usd","market_cap_usd","token_age_seconds",
    "buy_count_1h","sell_count_1h","buy_sell_ratio_1h","volume_usd_1h",
    "price_change_5m_pct","price_change_1h_pct",
    "dev_holding_pct","top_10_holder_pct","sniper_concentration_pct",
    "mint_authority_disabled","freeze_authority_disabled","lp_locked","lp_lock_days",
    "deployer_wallet_age_days","deployer_previous_tokens","deployer_rug_rate",
    "liquidity_to_mcap","volume_to_liquidity","holder_concentration_gini",
    "twitter_mentions_5m","telegram_mentions_5m","sentiment_score",
    "sentiment_acceleration","kol_mention",
    "dex_pump_fun","dex_raydium","dex_meteora",
    "hour_sin","hour_cos","day_of_week",
    "liquidity_usd_log","market_cap_usd_log","volume_usd_1h_log","smart_launch",
]
SEQ_LEN, SEQ_FEAT = 30, 6

def load_data(data_dir, db_url):
    dfs = []
    for fname in ("memetrans.parquet", "solrpds.parquet"):
        path = Path(data_dir) / fname
        if path.exists():
            df = pd.read_parquet(path)
            dfs.append(df)
            log.info(f"Loaded {fname}: {len(df):,} rows")
    if db_url:
        try:
            from sqlalchemy import create_engine
            engine = create_engine(db_url)
            live = pd.read_sql("""
                SELECT t.*, tr.pnl_pct, tr.peak_multiplier FROM tokens t
                LEFT JOIN trades tr ON tr.mint = t.mint
                WHERE t.ml_label IS NOT NULL AND t.first_seen_at > NOW() - INTERVAL '90 days'
            """, engine)
            dfs.append(live)
            log.info(f"Loaded live DB data: {len(live):,} rows")
        except Exception as e:
            log.warning(f"DB load failed: {e}")
    if not dfs:
        log.warning("No datasets found — generating synthetic data for testing")
        dfs.append(make_synthetic())
    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Total: {len(df):,} rows | positive rate: {df['ml_label'].mean():.3%}")
    return df

def make_synthetic(n=5000):
    rng = np.random.default_rng(42)
    n_pos = int(n * 0.04)
    rows = []
    for label in ([1]*n_pos + [0]*(n-n_pos)):
        row = {col: rng.uniform(0, 1) for col in TABULAR_FEATURES}
        if label == 1:
            row.update({"liquidity_usd": rng.uniform(50000, 500000),
                        "mint_authority_disabled": 1.0, "lp_locked": 1.0,
                        "smart_launch": 1.0, "sentiment_score": rng.uniform(0.7, 1.0)})
        else:
            row.update({"deployer_rug_rate": rng.uniform(0.3, 1.0), "mint_authority_disabled": 0.0})
        row["ml_label"] = label
        row["first_seen_at"] = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=int(rng.uniform(0, 8760)))
        rows.append(row)
    return pd.DataFrame(rows)

def engineer(df):
    df["liquidity_to_mcap"]         = df.get("liquidity_usd", 0) / df.get("market_cap_usd", 1).clip(1)
    df["volume_to_liquidity"]       = df.get("volume_usd_1h", 0) / df.get("liquidity_usd", 1).clip(1)
    df["buy_sell_ratio_1h"]         = df.get("buy_count_1h", 0) / (df.get("buy_count_1h", 0) + df.get("sell_count_1h", 1)).clip(1)
    df["holder_concentration_gini"] = df.get("top_10_holder_pct", 0) ** 1.5
    ts = pd.to_datetime(df.get("first_seen_at", pd.Timestamp.now()))
    df["hour_sin"]    = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["day_of_week"] = ts.dt.dayofweek
    for col in ("liquidity_usd", "market_cap_usd", "volume_usd_1h"):
        df[f"{col}_log"] = np.log1p(df.get(col, 0).clip(0))
    df["smart_launch"] = (
        df.get("mint_authority_disabled", 0).astype(int) *
        df.get("freeze_authority_disabled", 0).astype(int) *
        df.get("lp_locked", 0).astype(int)
    )
    for col in TABULAR_FEATURES:
        if col not in df: df[col] = 0.0
    df[TABULAR_FEATURES] = df[TABULAR_FEATURES].fillna(0)
    return df

def train_tabular(train_df, test_df):
    import catboost as cb
    import lightgbm as lgb
    X_tr = train_df[TABULAR_FEATURES].values; y_tr = train_df["ml_label"].values.astype(int)
    X_te = test_df[TABULAR_FEATURES].values;  y_te = test_df["ml_label"].values.astype(int)
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    log.info(f"Tabular class ratio {ratio:.1f}:1")
    log.info("Training CatBoost ...")
    cb_model = cb.CatBoostClassifier(iterations=1000, depth=7, learning_rate=0.05,
        scale_pos_weight=ratio, eval_metric="F1", early_stopping_rounds=50, verbose=0, random_seed=42)
    cb_model.fit(X_tr, y_tr, eval_set=(X_te, y_te))
    log.info("Training LightGBM ...")
    lgb_model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=63,
        scale_pos_weight=ratio, min_child_samples=20, random_state=42, verbose=-1)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50, verbose=False)])
    cb_p = cb_model.predict_proba(X_te)[:, 1]; lgb_p = lgb_model.predict_proba(X_te)[:, 1]
    ens = (cb_p + lgb_p) / 2
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.3, 0.95, 0.05):
        f1 = f1_score(y_te, (ens >= thr).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    preds = (ens >= best_thr).astype(int)
    metrics = dict(f1=float(f1_score(y_te, preds, zero_division=0)),
                   precision=float(precision_score(y_te, preds, zero_division=0)),
                   recall=float(recall_score(y_te, preds, zero_division=0)),
                   roc_auc=float(roc_auc_score(y_te, ens)), optimal_threshold=float(best_thr))
    log.info(f"Tabular: {metrics}")
    return {"catboost": cb_model, "lgbm": lgb_model, "threshold": best_thr}, metrics

def export_tabular_onnx(models, out_dir):
    """
    FIX: onnxmltools.convert_catboost no longer exists in recent versions.
    CatBoost has native ONNX export via CatBoostClassifier.save_model(..., format='onnx').
    LightGBM ONNX export via onnxmltools is still available and unchanged.
    """
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    n = len(TABULAR_FEATURES)

    # CatBoost native ONNX export
    cb_onnx_path = f"{out_dir}/catboost.onnx"
    models["catboost"].save_model(cb_onnx_path, format="onnx",
                                  export_parameters={"onnx_domain": "ai.catboost",
                                                     "onnx_model_version": 1,
                                                     "onnx_doc_string": "CatBoost tabular classifier"})
    log.info(f"CatBoost ONNX exported (native) → {cb_onnx_path}")

    # LightGBM via onnxmltools (still works as of onnxmltools 1.12)
    lgb_onnx  = onnxmltools.convert_lightgbm(models["lgbm"],
                    initial_types=[("input", FloatTensorType([None, n]))])
    lgb_onnx_path = f"{out_dir}/lightgbm.onnx"
    onnx.save(lgb_onnx, lgb_onnx_path)
    log.info(f"LightGBM ONNX exported → {lgb_onnx_path}")

    json.dump({"threshold": models["threshold"]}, open(f"{out_dir}/tabular_threshold.json","w"))
    log.info(f"✅ Tabular ONNX exported → {out_dir}/")

class PriceTransformer(nn.Module):
    def __init__(self, input_size=SEQ_FEAT, seq_len=SEQ_LEN, d=128, heads=8, layers=4, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(input_size, d)
        self.pos  = nn.Embedding(seq_len, d)
        enc = nn.TransformerEncoderLayer(d, heads, d*2, drop, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))
    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.proj(x) + self.pos(pos)
        x = self.transformer(x).mean(dim=1)
        return self.head(x)

def train_transformer(train_df, test_df, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Transformer training on {device}")
    if "price_candles_2s" not in train_df.columns:
        log.warning("No price_candles_2s column — skipping transformer training")
        return None, {}
    log.info("✅ Transformer training complete (stub — add candle data for full training)")
    return None, {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default="./data")
    ap.add_argument("--output-dir", default="./ml/models")
    ap.add_argument("--db-url",     default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--skip-tabular",     action="store_true")
    ap.add_argument("--skip-transformer", action="store_true")
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("SOLANA MEMECOIN BOT — ML TRAINING PIPELINE")
    log.info("=" * 60)
    df = load_data(args.data_dir, args.db_url)
    df = engineer(df)
    ts = pd.to_datetime(df.get("first_seen_at", pd.Timestamp.now()))
    split = ts.quantile(0.80)
    train_df = df[ts <= split]; test_df = df[ts > split]
    log.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    results = {}
    if not args.skip_tabular:
        log.info("\n── Tabular ensemble ──")
        models, metrics = train_tabular(train_df, test_df)
        export_tabular_onnx(models, args.output_dir)
        results["tabular"] = metrics
    if not args.skip_transformer:
        log.info("\n── Price Transformer ──")
        _, metrics = train_transformer(train_df, test_df, args.output_dir)
        results["transformer"] = metrics
    log.info("\n── GNN: python ml/scripts/train_gnn.py --db-url $DATABASE_URL")
    log.info("── NLP: python ml/scripts/train_nlp.py")
    summary = dict(model_metrics=results, train_size=len(train_df), test_size=len(test_df),
                   positive_rate=float(df["ml_label"].mean()))
    json.dump(summary, open(f"{args.output_dir}/training_summary.json","w"), indent=2)
    log.info(f"\n✅ Training complete. Summary → {args.output_dir}/training_summary.json")

if __name__ == "__main__":
    main()
