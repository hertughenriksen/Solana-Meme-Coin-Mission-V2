"""train_nlp.py — Fine-tunes ProsusAI/finbert for crypto social sentiment.
Usage: python ml/scripts/train_nlp.py --output-dir ./ml/models
"""
import argparse, json, logging, os, re
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_MODEL="ProsusAI/finbert"; MAX_LEN=128; BATCH=32; LR=2e-5; EPOCHS=8; NUM_LABELS=3

EXAMPLES = [
    ("just launched on pump.fun LP locked 30 days mint disabled based dev", 2),
    ("fresh launch CA: [MINT] only 500k mcap massive buy pressure", 2),
    ("smart money buying [MINT] liquidity locked sniper count low", 2),
    ("pump alert CA [MINT] huge buys coming in early alpha call", 2),
    ("massive KOL signal on [MINT] dev renounced contract", 2),
    ("what do you think about solana memecoins this week", 1),
    ("sol price holding 150 support good for memecoins generally", 1),
    ("not financial advice just sharing my research on the space", 1),
    ("dev just sold 50% of supply dumping hard [MINT] avoid", 0),
    ("rug alert [MINT] liquidity pulled run run run", 0),
    ("honeypot [MINT] cant sell contract blocked", 0),
    ("scam token [MINT] same deployer as 5 previous rugs", 0),
    ("mint authority still enabled on [MINT] infinite print risk", 0),
    ("sniper bots bought 40% supply first block [MINT] exit scam", 0),
]

CRYPTO_VOCAB = {"lp":"liquidity pool","mc":"market cap","mcap":"market cap",
                "devs":"developers","kol":"key opinion leader","ape":"buy aggressively",
                "aped":"bought aggressively","rugged":"rug pulled","lfg":"lets go"}

def preprocess(text):
    text = re.sub(r"[1-9A-HJ-NP-Za-km-z]{43,44}","[MINT]",text)
    text = re.sub(r"https?://\S+","[URL]",text)
    text = re.sub(r"\$[A-Z]{2,10}","[TICKER]",text)
    text = text.lower()
    for abbr,exp in CRYPTO_VOCAB.items(): text=re.sub(r""+abbr+r"",exp,text)
    return " ".join(text.split())[:512]

class CryptoSentimentDataset(Dataset):
    def __init__(self,texts,labels,tokenizer):
        enc=tokenizer([preprocess(t) for t in texts],max_length=MAX_LEN,padding="max_length",truncation=True,return_tensors="pt")
        self.ids=enc["input_ids"]; self.mask=enc["attention_mask"]
        self.ttids=enc.get("token_type_ids",torch.zeros_like(enc["input_ids"]))
        self.labels=torch.tensor(labels,dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        return {"input_ids":self.ids[i],"attention_mask":self.mask[i],"token_type_ids":self.ttids[i],"labels":self.labels[i]}

class CryptoSentimentModel(nn.Module):
    def __init__(self,base=BASE_MODEL,drop=0.2):
        super().__init__()
        self.bert=BertModel.from_pretrained(base)
        h=self.bert.config.hidden_size
        self.drop=nn.Dropout(drop)
        self.cls=nn.Sequential(nn.Linear(h,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,NUM_LABELS))
        self.vel=nn.Sequential(nn.Linear(h,64),nn.GELU(),nn.Linear(64,1),nn.Tanh())
    def forward(self,input_ids,attention_mask,token_type_ids=None):
        out=self.bert(input_ids,attention_mask,token_type_ids)
        cls=self.drop(out.last_hidden_state[:,0,:])
        return self.cls(cls),self.vel(cls).squeeze(-1)

def augment(texts,labels):
    from collections import Counter
    cnt=Counter(labels); mx=max(cnt.values()); at,al=list(texts),list(labels)
    for lbl,c in cnt.items():
        examples=[(t,l) for t,l in zip(texts,labels) if l==lbl]
        for i in range(mx-c):
            t,_=examples[i%len(examples)]; words=t.split()
            if len(words)>3: words.pop(np.random.randint(len(words)))
            at.append(" ".join(words)); al.append(lbl)
    combined=list(zip(at,al)); np.random.shuffle(combined); return zip(*combined)

def train_model(output_dir, db_url=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"NLP training on {device}")
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL)
    texts,labels=[t for t,_ in EXAMPLES],[l for _,l in EXAMPLES]
    texts,labels=augment(texts,list(labels)); texts,labels=list(texts),list(labels)
    split=int(len(texts)*0.8)
    tr_ds=CryptoSentimentDataset(texts[:split],labels[:split],tokenizer)
    va_ds=CryptoSentimentDataset(texts[split:],labels[split:],tokenizer)
    tr_ld=DataLoader(tr_ds,BATCH,shuffle=True); va_ld=DataLoader(va_ds,BATCH)
    model=CryptoSentimentModel().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    total=len(tr_ld)*EPOCHS; sched=get_linear_schedule_with_warmup(opt,int(total*0.1),total)
    cls_crit=nn.CrossEntropyLoss(); best_f1,best_state,patience=0,None,0
    for epoch in range(1,EPOCHS+1):
        model.train(); tr_loss=0
        for batch in tr_ld:
            opt.zero_grad()
            ids=batch["input_ids"].to(device); mask=batch["attention_mask"].to(device)
            ttids=batch["token_type_ids"].to(device); lbls=batch["labels"].to(device)
            logits,vel=model(ids,mask,ttids)
            loss=cls_crit(logits,lbls)+0.1*F.mse_loss(vel,lbls.float()-1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step(); tr_loss+=loss.item()
        model.eval(); all_p,all_l=[],[]
        with torch.no_grad():
            for batch in va_ld:
                ids=batch["input_ids"].to(device); mask=batch["attention_mask"].to(device)
                ttids=batch["token_type_ids"].to(device)
                logits,_=model(ids,mask,ttids)
                all_p.extend(logits.argmax(-1).cpu().tolist()); all_l.extend(batch["labels"].tolist())
        f1=f1_score(all_l,all_p,average="macro",zero_division=0)
        if f1>best_f1: best_f1,best_state,patience=f1,{k:v.clone() for k,v in model.state_dict().items()},0
        else: patience+=1
        log.info(f"Epoch {epoch} | loss={tr_loss/len(tr_ld):.4f} | F1={f1:.4f}")
        if patience>=3: break
    model.load_state_dict(best_state); model.cpu().eval()
    class SentimentWrapper(nn.Module):
        def __init__(self,m): super().__init__(); self.m=m
        def forward(self,ids,mask,ttids):
            logits,vel=self.m(ids,mask,ttids); probs=F.softmax(logits,-1)
            return probs[:,0]*0+probs[:,1]*0.5+probs[:,2]*1.0, vel
    wrapper=SentimentWrapper(model).eval()
    dummy_ids=torch.zeros(1,MAX_LEN,dtype=torch.long)
    dummy_mask=torch.ones(1,MAX_LEN,dtype=torch.long)
    dummy_tt=torch.zeros(1,MAX_LEN,dtype=torch.long)
    onnx_path=f"{output_dir}/sentiment_finbert.onnx"
    torch.onnx.export(wrapper,(dummy_ids,dummy_mask,dummy_tt),onnx_path,opset_version=17,
                      input_names=["input_ids","attention_mask","token_type_ids"],
                      output_names=["sentiment_score","velocity"],
                      dynamic_axes={k:{0:"batch"} for k in ["input_ids","attention_mask","token_type_ids","sentiment_score","velocity"]})
    tokenizer.save_pretrained(f"{output_dir}/finbert_tokenizer")
    with open(f"{output_dir}/nlp_metrics.json","w") as f: json.dump({"f1_macro":float(best_f1)},f,indent=2)
    log.info(f"NLP complete | F1={best_f1:.4f} | ONNX -> {onnx_path}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default="./ml/models")
    parser.add_argument("--db-url",default=os.environ.get("DATABASE_URL"))
    args=parser.parse_args()
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    train_model(args.output_dir,args.db_url)

if __name__=="__main__": main()
