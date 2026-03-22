"""train_gnn.py — Temporal Graph Attention Network for rug pull detection.
Usage: python ml/scripts/train_gnn.py --synthetic
       python ml/scripts/train_gnn.py --db-url $DATABASE_URL
"""
import argparse, json, logging, os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.metrics import f1_score, roc_auc_score, recall_score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

NODE_DIM=12; EDGE_DIM=4; HIDDEN=128; HEADS=4; LAYERS=3; DROPOUT=0.3
LR=3e-4; BATCH=64; EPOCHS=80; PATIENCE=12

def node_features(wallet):
    return np.array([
        np.log1p(wallet.get("wallet_age_days", 0)),
        np.log1p(wallet.get("total_transactions", 0)),
        np.log1p(wallet.get("total_sol_volume", 0)),
        min(wallet.get("rug_involvement_count", 0), 10) / 10.0,
        float(wallet.get("is_deployer", False)),
        float(wallet.get("is_known_sniper", False)),
        np.log1p(wallet.get("sol_balance_at_launch", 0)),
        np.log1p(wallet.get("num_tokens_held", 0)),
        np.log1p(wallet.get("avg_hold_duration_hours", 0)),
        wallet.get("win_rate", 0.5),
        float(wallet.get("funded_by_exchange", False)),
        np.log1p(wallet.get("hours_since_last_activity", 0)),
    ], dtype=np.float32)

def edge_features(tx, launch_time):
    delta = (tx.get("timestamp", launch_time) - launch_time) / 60.0
    return np.array([np.log1p(tx.get("sol_amount", 0)), np.clip(delta/30.0,0,1),
                     float(tx.get("is_same_bundle", False)), float(tx.get("is_first_block", False))], dtype=np.float32)

class WalletGATN(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(NODE_DIM, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())
        self.edge_enc = nn.Sequential(nn.Linear(EDGE_DIM, HIDDEN//2), nn.GELU())
        self.gat   = nn.ModuleList([GATConv(HIDDEN, HIDDEN//HEADS, heads=HEADS, edge_dim=HIDDEN//2, dropout=DROPOUT, concat=True) for _ in range(LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(LAYERS)])
        self.freq  = nn.Sequential(nn.Linear(HIDDEN*2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN//2), nn.GELU())
        self.cls   = nn.Sequential(nn.Linear(HIDDEN//2, 64), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(64, 1))

    def forward(self, data):
        x  = self.node_enc(data.x)
        ea = self.edge_enc(data.edge_attr)
        b  = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for gat, norm in zip(self.gat, self.norms):
            x = norm(x + F.dropout(gat(x, data.edge_index, edge_attr=ea), DROPOUT, self.training))
        x = self.freq(torch.cat([global_mean_pool(x, b), global_max_pool(x, b)], -1))
        return torch.sigmoid(self.cls(x)).squeeze(-1)

def synthetic_graphs(n_rug=1500, n_legit=1500):
    graphs = []
    for label, n_graphs in [(1, n_rug), (0, n_legit)]:
        for _ in range(n_graphs):
            n  = np.random.randint(3, 20)
            x  = np.random.randn(n, NODE_DIM).astype(np.float32)
            if label == 1:
                x[0,0]=np.log1p(1); x[0,4]=1.0; x[0,3]=np.random.uniform(0.5,1.0)
                for j in range(1, min(n-1, np.random.randint(3,8))+1): x[j,5]=1.0
            else:
                x[0,0]=np.log1p(np.random.uniform(30,500)); x[0,4]=1.0; x[0,3]=0.0
            n_e=n*2; src=np.random.randint(0,n,n_e); dst=np.random.randint(0,n,n_e)
            ea=np.random.randn(n_e, EDGE_DIM).astype(np.float32)
            if label==1: ea[:min(5,n_e),2]=1.0
            graphs.append(Data(x=torch.tensor(x), edge_index=torch.tensor([src,dst],dtype=torch.long),
                               edge_attr=torch.tensor(ea), y=torch.tensor([label]), num_nodes=n))
    np.random.shuffle(graphs)
    return graphs

def train(model, loader, opt, device):
    model.train(); loss_sum=0; crit=nn.BCELoss()
    for batch in loader:
        batch=batch.to(device); opt.zero_grad()
        loss=crit(model(batch), batch.y.float())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        loss_sum+=loss.item()
    return loss_sum/len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); probs,labels=[],[]
    for batch in loader:
        batch=batch.to(device)
        probs.extend(model(batch).cpu().tolist()); labels.extend(batch.y.cpu().tolist())
    probs,labels=np.array(probs),np.array(labels)
    preds=(probs>=0.5).astype(int)
    return dict(f1=f1_score(labels,preds,zero_division=0),
                auc=roc_auc_score(labels,probs) if len(set(labels))>1 else 0.5,
                rug_recall=recall_score(labels,preds,zero_division=0))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./ml/models")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"))
    args=parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"GNN training on {device}")
    graphs=synthetic_graphs(2000,2000)
    split=int(len(graphs)*0.8)
    tr_loader=DataLoader(graphs[:split],BATCH,shuffle=True)
    va_loader=DataLoader(graphs[split:],BATCH)
    model=WalletGATN().to(device)
    opt=torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, 1e-6)
    best_f1,best_state,patience=0,None,0
    for epoch in range(1, EPOCHS+1):
        loss=train(model,tr_loader,opt,device); m=evaluate(model,va_loader,device); sched.step()
        if m["f1"]>best_f1:
            best_f1,best_state,patience=m["f1"],{k:v.clone() for k,v in model.state_dict().items()},0
            torch.save(best_state, f"{args.output_dir}/wallet_gnn_best.pt")
        else: patience+=1
        if patience>=PATIENCE: log.info(f"Early stop epoch {epoch}"); break
        if epoch%10==0: log.info(f"Epoch {epoch:3d} | loss={loss:.4f} | F1={m["f1"]:.4f} | AUC={m["auc"]:.4f}")
    model.load_state_dict(best_state); final=evaluate(model,va_loader,device)
    log.info(f"Final | F1={final["f1"]:.4f} | AUC={final["auc"]:.4f} | rug_recall={final["rug_recall"]:.4f}")
    class Wrapper(nn.Module):
        def __init__(self,m): super().__init__(); self.enc=m.node_enc; self.freq=m.freq; self.cls=m.cls
        def forward(self,x): h=self.enc(x); h=self.freq(torch.cat([h,h],-1)); return torch.sigmoid(self.cls(h))
    model.cpu().eval(); wrapper=Wrapper(model).eval(); dummy=torch.zeros(1,NODE_DIM)
    torch.onnx.export(wrapper,dummy,f"{args.output_dir}/wallet_gnn.onnx",opset_version=17,
                      input_names=["node_features"],output_names=["rug_probability"],
                      dynamic_axes={"node_features":{0:"batch"},"rug_probability":{0:"batch"}})
    torch.save(model.state_dict(), f"{args.output_dir}/wallet_gnn_full.pt")
    with open(f"{args.output_dir}/gnn_metrics.json","w") as f: json.dump(final,f,indent=2)
    log.info(f"GNN complete — ONNX + full weights saved to {args.output_dir}/")

if __name__=="__main__": main()
