"""gnn_sidecar.py — Real-time GNN scoring service.
Reads TokenSignal graph data from Redis, runs TGAT inference,
writes rug probability score back for the Rust bot to read.
Start: python ml/scripts/gnn_sidecar.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio, json, logging
import numpy as np
import torch
import redis.asyncio as aioredis

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

REDIS_URL    = os.environ.get('REDIS_URL', 'redis://localhost:6379')
MODEL_PATH   = os.environ.get('GNN_MODEL_PATH', './ml/models/wallet_gnn_full.pt')
SIGNAL_CHAN  = 'gnn:signals'
SCORE_PREFIX = 'gnn:score:'
SCORE_TTL    = 300
NODE_DIM = 12; EDGE_DIM = 4

async def main():
    from train_gnn import WalletGATN, node_features, edge_features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WalletGATN().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        log.info(f"GNN sidecar loaded model from {MODEL_PATH}")
    else:
        log.warning(
            f"Model not found at {MODEL_PATH} — using random weights "
            f"(run train_gnn.py first)"
        )
    model.eval()
    r = aioredis.from_url(REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe(SIGNAL_CHAN)
    log.info(f"GNN sidecar ready on {device} | listening on Redis channel '{SIGNAL_CHAN}'")
    async for message in pubsub.listen():
        if message['type'] != 'message': continue
        try:
            payload = json.loads(message['data'])
            mint = payload.get('mint', '')
            if not mint: continue
            graph_data = payload.get('graph_data', {})
            from train_gnn import node_features as nf, edge_features as ef
            score = score_signal(model, device, graph_data, mint, nf, ef)
            await r.setex(f"{SCORE_PREFIX}{mint}", SCORE_TTL, str(score))
            log.debug(f"Scored {mint[:8]} -> {score:.3f}")
        except Exception as e:
            log.error(f"GNN sidecar error: {e}")

def score_signal(model, device, graph_data, mint, node_features, edge_features):
    from torch_geometric.data import Data, Batch
    wallets      = graph_data.get('wallets', {})
    transactions = graph_data.get('transactions', [])
    launch_time  = graph_data.get('launch_timestamp', 0)
    if not wallets:
        feats = np.zeros((1, NODE_DIM), dtype=np.float32); feats[0,4]=1.0
        x = torch.tensor(feats)
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        ea = torch.zeros(1, EDGE_DIM)
        g = Data(x=x, edge_index=edge_index, edge_attr=ea, num_nodes=1)
    else:
        wallet_list = list(wallets.keys()); w2i={w:i for i,w in enumerate(wallet_list)}
        x = np.stack([node_features(wallets[w]) for w in wallet_list])
        src,dst,eas=[],[],[]
        for tx in transactions:
            s=tx.get('from'); d=tx.get('to')
            if s in w2i and d in w2i:
                src.append(w2i[s]); dst.append(w2i[d]); eas.append(edge_features(tx,launch_time))
        if not src: src,dst=[0],[0]; eas=[np.zeros(EDGE_DIM,dtype=np.float32)]
        g = Data(x=torch.tensor(x), edge_index=torch.tensor([src,dst],dtype=torch.long),
                 edge_attr=torch.tensor(np.stack(eas)), num_nodes=len(wallet_list))
    batch = Batch.from_data_list([g]).to(device)
    with torch.no_grad(): rug_prob = model(batch).item()
    return 1.0 - rug_prob

if __name__ == '__main__':
    asyncio.run(main())
