"""
train_gnn.py — Graph Neural Network training for deployer-wallet rug detection.

BUG FIX: Python ≤ 3.11 raises SyntaxError on f-strings that contain dict access
with the same quote style as the outer string, e.g. f"...{m["f1"]:.4f}...".
All such occurrences are replaced with a temporary variable pattern:
    val = m["f1"]; log.info(f"...{val:.4f}...")
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

log = logging.getLogger("train_gnn")


# ── Feature engineering ────────────────────────────────────────────────────────

def node_features(wallet: dict) -> list[float]:
    """12-element node feature vector (must match models/mod.rs:build_deployer_node_features)."""
    rug  = wallet.get("rug_involvement", 0)
    tot  = wallet.get("total_transactions", 1)
    n_tok = wallet.get("num_tokens_held", 0)
    win  = wallet.get("win_rate", 0.5)
    return [
        np.log1p(wallet.get("wallet_age_days", 0)),
        np.log1p(wallet.get("total_transactions", 0)),
        np.log1p(wallet.get("total_sol_volume", 0)),
        min(rug, 10) / 10.0,
        float(wallet.get("is_deployer", False)),
        float(wallet.get("is_known_sniper", False)),
        np.log1p(wallet.get("sol_balance_at_launch", 0)),
        np.log1p(n_tok),
        np.log1p(wallet.get("avg_hold_duration_hours", 0)),
        win,
        float(wallet.get("funded_by_exchange", False)),
        np.log1p(wallet.get("hours_since_last_activity", 0)),
    ]


def build_graph(token: dict) -> Data:
    """Convert one token's wallet graph to a PyG Data object."""
    wallets    = token["wallets"]          # list of wallet dicts
    adj        = token["adjacency"]        # list of [src_idx, dst_idx] edges
    deployer_i = token.get("deployer_index", 0)

    x = torch.tensor([node_features(w) for w in wallets], dtype=torch.float)
    if adj:
        edge_index = torch.tensor(adj, dtype=torch.long).t().contiguous()
    else:
        # Self-loop only (isolated deployer wallet)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    # Global rug label for the whole graph (0 = safe, 1 = rug)
    y = torch.tensor([int(token.get("is_rug", 0))], dtype=torch.float)

    # Deployer node index used for pool-then-index prediction head
    deployer = torch.tensor([deployer_i], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y, deployer=deployer)


# ── Model ─────────────────────────────────────────────────────────────────────

class WalletGNN(torch.nn.Module):
    def __init__(self, in_dim: int = 12, hidden: int = 64, n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs   = torch.nn.ModuleList()
        self.bns     = torch.nn.ModuleList()
        dims = [in_dim] + [hidden] * n_layers
        for i in range(n_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(torch.nn.BatchNorm1d(dims[i + 1]))
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 32), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x, edge_index, batch, deployer):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Use deployer node embedding rather than mean-pool (more informative)
        deployer_emb = x[deployer]               # shape [batch_size, hidden]
        return self.head(deployer_emb).squeeze(-1)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(db_url: str) -> list[Data]:
    conn = psycopg2.connect(db_url)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            dw.wallet,
            dw.rug_count,
            dw.total_tokens_deployed,
            dw.wallet_age_days,
            dw.win_rate,
            array_agg(row_to_json(t.*)) AS token_list,
            (dw.rug_count::float / GREATEST(dw.total_tokens_deployed, 1)) > 0.4 AS is_rug
        FROM deployer_wallets dw
        JOIN tokens t ON t.deployer_wallet = dw.wallet
        WHERE t.ml_label IS NOT NULL
        GROUP BY dw.wallet, dw.rug_count, dw.total_tokens_deployed,
                 dw.wallet_age_days, dw.win_rate
        HAVING COUNT(t.*) >= 2
    """)
    rows = cur.fetchall()
    conn.close()

    graphs = []
    for row in rows:
        # Build a minimal ego-graph: deployer + top holders as satellite nodes
        tokens = row["token_list"] or []
        deployer_node = {
            "wallet_age_days":        row.get("wallet_age_days", 0),
            "total_transactions":     (row.get("total_tokens_deployed", 0) or 0) * 10,
            "total_sol_volume":       0,
            "rug_involvement":        row.get("rug_count", 0),
            "is_deployer":            True,
            "is_known_sniper":        False,
            "sol_balance_at_launch":  0,
            "num_tokens_held":        len(tokens),
            "avg_hold_duration_hours": 0,
            "win_rate":               row.get("win_rate", 0.5) or 0.5,
            "funded_by_exchange":     False,
            "hours_since_last_activity": 0,
        }
        wallets = [deployer_node]
        # Add satellite nodes from top_holders JSON stored on the token row
        for tok in tokens[:5]:
            holders = tok.get("top_holders_json") or []
            for h in (holders[:2] if isinstance(holders, list) else []):
                if isinstance(h, dict):
                    wallets.append({
                        "wallet_age_days":       h.get("age_days", 30),
                        "total_transactions":    100,
                        "total_sol_volume":      h.get("sol_volume", 0),
                        "rug_involvement":       0,
                        "is_deployer":           False,
                        "is_known_sniper":       h.get("is_sniper", False),
                        "sol_balance_at_launch": 0,
                        "num_tokens_held":       1,
                        "avg_hold_duration_hours": 0,
                        "win_rate":              0.5,
                        "funded_by_exchange":    False,
                        "hours_since_last_activity": 24,
                    })

        n = len(wallets)
        # Fully-connect satellite nodes to deployer (star topology)
        adj = [[0, i] for i in range(1, n)] + [[i, 0] for i in range(1, n)]

        is_rug = bool(row.get("is_rug", False))
        try:
            graphs.append(build_graph({
                "wallets": wallets, "adjacency": adj, "deployer_index": 0, "is_rug": is_rug,
            }))
        except Exception as exc:
            log.warning("Skipping row for wallet %s: %s", row["wallet"], exc)

    log.info("Loaded %d wallet graphs (%d rugs)", len(graphs), sum(g.y.item() > 0 for g in graphs))
    return graphs


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optim.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch, batch.deployer)
        loss   = F.binary_cross_entropy_with_logits(
            logits, batch.y,
            pos_weight=torch.tensor([3.0], device=device),   # rugs are rare
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch, batch.deployer)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds.extend(probs.tolist())
        labels.extend(batch.y.cpu().numpy().tolist())

    preds_b = [1 if p >= 0.5 else 0 for p in preds]
    tp = sum(p == 1 and l == 1 for p, l in zip(preds_b, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds_b, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds_b, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds_b, labels))

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    acc       = (tp + tn) / max(len(labels), 1)

    # FIX (Bug #1): avoid same-quote dict access inside f-strings.
    # Store values in locals before interpolating.
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def export_onnx(model, out_path: str, device):
    """Export the GNN as ONNX. Uses a fixed-size single-node dummy input."""
    model.eval()
    dummy_x          = torch.zeros((1, 12), device=device)
    dummy_edge_index = torch.zeros((2, 1), dtype=torch.long, device=device)
    dummy_batch      = torch.zeros(1, dtype=torch.long, device=device)
    dummy_deployer   = torch.zeros(1, dtype=torch.long, device=device)

    class OnnxWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, node_features):
            # Simplified: single-node inference without edges (deployer only)
            x = node_features
            for conv, bn in zip(self.inner.convs, self.inner.bns):
                # Without edges we just apply the linear transform part
                x = conv.lin(x)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.0, training=False)
            return torch.sigmoid(self.inner.head(x).squeeze(-1))

    wrapper = OnnxWrapper(model).to(device)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_x,),
        out_path,
        input_names=["node_features"],
        output_names=["rug_probability"],
        dynamic_axes={"node_features": {0: "batch_size"}, "rug_probability": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    log.info("ONNX model exported to %s", out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url",     required=True)
    parser.add_argument("--output-dir", default="./ml/models")
    parser.add_argument("--epochs",     type=int, default=80)
    parser.add_argument("--hidden",     type=int, default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout",    type=float, default=0.3)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = load_data(args.db_url)
    if len(graphs) < 20:
        log.warning("Only %d graphs — need more wallet data; skipping GNN training", len(graphs))
        return

    # Train / val split (80/20)
    n_train  = int(len(graphs) * 0.8)
    train_ds = graphs[:n_train]
    val_ds   = graphs[n_train:]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  follow_batch=["deployer"])
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, follow_batch=["deployer"])

    model = WalletGNN(in_dim=12, hidden=args.hidden, dropout=args.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)

    best_f1     = 0.0
    best_path   = str(out_dir / "gnn_best.pt")
    patience    = 15
    no_improve  = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, device)
        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            m   = evaluate(model, val_loader, device)
            # FIX (Bug #1): extract values before f-string interpolation.
            f1_val   = m["f1"]
            acc_val  = m["accuracy"]
            rec_val  = m["recall"]
            prec_val = m["precision"]
            log.info(
                "Epoch %3d | loss %.4f | F1 %.4f | acc %.4f | rec %.4f | prec %.4f",
                epoch, train_loss, f1_val, acc_val, rec_val, prec_val,
            )
            if f1_val > best_f1:
                best_f1    = f1_val
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                log.info("  ✅ New best F1 %.4f — checkpoint saved", best_f1)
            else:
                no_improve += 5
                if no_improve >= patience:
                    log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                    break

    # Reload best and export
    model.load_state_dict(torch.load(best_path, map_location=device))
    onnx_path = str(out_dir / "gnn.onnx")
    export_onnx(model, onnx_path, device)

    # Save metrics
    final_metrics = evaluate(model, val_loader, device)
    metrics_path  = str(out_dir / "gnn_metrics.json")
    with open(metrics_path, "w") as f:
        # FIX (Bug #1): no same-quote dict access in f-strings — use json.dump directly.
        json.dump({"best_val_f1": best_f1, **final_metrics}, f, indent=2)

    # Final summary log — FIX (Bug #1): temp vars for all dict values.
    fm_f1   = final_metrics["f1"]
    fm_acc  = final_metrics["accuracy"]
    fm_rec  = final_metrics["recall"]
    fm_prec = final_metrics["precision"]
    log.info(
        "GNN training complete | F1 %.4f | acc %.4f | recall %.4f | precision %.4f",
        fm_f1, fm_acc, fm_rec, fm_prec,
    )
    log.info("ONNX → %s | Metrics → %s", onnx_path, metrics_path)


if __name__ == "__main__":
    main()
