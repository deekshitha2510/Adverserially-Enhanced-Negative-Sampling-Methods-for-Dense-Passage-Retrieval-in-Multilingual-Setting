# ============================================================
#                 FINAL AXL-ICT TRAINING PIPELINE
# ============================================================

!pip install -q faiss-cpu sentence-transformers

import os, json, random, math
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.cluster import MiniBatchKMeans

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = "/content/final_axl_ict"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MSMARCO_PATH = "/content/msmarco_triples.csv"
TYDI_DIR     = "/content"  # your filtered negatives directory

MSM_SAMPLES  = 1000
TYDI_SAMPLES = 1100
BATCH_SIZE   = 8
EPOCHS       = 10
MAX_LEN      = 64


# ---------------------------
# LOAD MSMARCO
# ---------------------------
df = pd.read_csv(MSMARCO_PATH, engine="python", on_bad_lines="skip").sample(MSM_SAMPLES)
msm = []
for _,r in df.iterrows():
    msm.append({
        "q":str(r["query"]),
        "pos":str(r["positive"]),
        "neg":str(r["negative"])
    })


# ---------------------------
# LOAD FILTERED TYDI
# ---------------------------
def load_jsonl(path):
    out=[]
    with open(path,"r",encoding="utf8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except:
                pass
    return out

files = sorted([f for f in os.listdir(TYDI_DIR) if f.endswith(".jsonl")])
tydi = []

take_per_lang = max(1, TYDI_SAMPLES // len(files))

for f in files:
    rows = load_jsonl(os.path.join(TYDI_DIR,f))[:take_per_lang]
    for r in rows:
        if "filtered_negatives_with_scores" in r and len(r["filtered_negatives_with_scores"])>0:
            neg = r["filtered_negatives_with_scores"][0]["neg"]
        else:
            neg = "This is an incorrect and irrelevant answer."
        tydi.append({
            "q": r["query"],
            "pos": r["positive_passage"],
            "neg": neg
        })

print("Loaded MSMARCO:", len(msm))
print("Loaded TYDI:", len(tydi))
train_data = msm + tydi
print("Total train triples:", len(train_data))


# ---------------------------
# DPR Model
# ---------------------------
class DPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_enc = AutoModel.from_pretrained(MODEL_NAME)
        self.p_enc = AutoModel.from_pretrained(MODEL_NAME)

    def encode_q(self, ids, mask):
        out = self.q_enc(input_ids=ids, attention_mask=mask).last_hidden_state
        return (out*mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)

    def encode_p(self, ids, mask):
        out = self.p_enc(input_ids=ids, attention_mask=mask).last_hidden_state
        return (out*mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DPR().to(DEVICE)
opt = AdamW(model.parameters(), lr=1e-5)


# ---------------------------
# Encode helper
# ---------------------------
def encode(model, texts, batch=64):
    embs=[]
    for i in range(0,len(texts),batch):
        t = tokenizer(texts[i:i+batch], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model.encode_p(t["input_ids"], t["attention_mask"])
        embs.append(o.cpu().numpy())
    return np.vstack(embs)


# ---------------------------
# ICT-P TRAINING LOOP
# ---------------------------
for epoch in range(1,EPOCHS+1):
    print(f"\nEPOCH {epoch}")

    # 1. cluster positive passages (ICT-P)
    pos_texts = [t["pos"] for t in train_data]
    pos_emb = encode(model, pos_texts, batch=64)
    k = max(1, len(pos_texts)//BATCH_SIZE)
    km = MiniBatchKMeans(n_clusters=k, batch_size=2048)
    km.fit(pos_emb)
    labels = km.labels_

    # create batches by cluster
    batches=[]
    cluster_map={}
    for i,l in enumerate(labels):
        cluster_map.setdefault(l,[]).append(i)

    for _, idxs in cluster_map.items():
        random.shuffle(idxs)
        for i in range(0,len(idxs),BATCH_SIZE):
            chunk = idxs[i:i+BATCH_SIZE]
            if len(chunk)>=2:
                batches.append(chunk)

    random.shuffle(batches)
    print("Batches:", len(batches))


    # 2. DPR loss training
    total_loss=0
    for batch in tqdm(batches):
        qs=[train_data[i]["q"] for i in batch]
        ps=[train_data[i]["pos"] for i in batch]
        ng=[train_data[i]["neg"] for i in batch]

        qtok=tokenizer(qs, padding=True, truncation=True,max_length=MAX_LEN,return_tensors="pt").to(DEVICE)
        ptok=tokenizer(ps, padding=True, truncation=True,max_length=MAX_LEN,return_tensors="pt").to(DEVICE)
        ntok=tokenizer(ng, padding=True, truncation=True,max_length=MAX_LEN,return_tensors="pt").to(DEVICE)

        q_emb = model.encode_q(qtok["input_ids"], qtok["attention_mask"])
        p_emb = model.encode_p(ptok["input_ids"], ptok["attention_mask"])
        n_emb = model.encode_p(ntok["input_ids"], ntok["attention_mask"])

        # DPR contrastive
        sim_pos = (q_emb*p_emb).sum(1)
        sim_neg = (q_emb*n_emb).sum(1)

        logits = torch.stack([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(len(batch), dtype=torch.long).to(DEVICE)

        loss = nn.CrossEntropyLoss()(logits,labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss+=loss.item()

    print("Epoch loss:", total_loss/len(batches))


# ---------------------------
# EVALUATION: MRR / Recall
# ---------------------------
queries  = [x["q"] for x in train_data]
gold_ids = list(range(len(train_data)))

# encode passages
corpus_emb = encode(model, [t["pos"] for t in train_data])
faiss.normalize_L2(corpus_emb)
index = faiss.IndexFlatIP(corpus_emb.shape[1])
index.add(corpus_emb)

# encode queries
q_emb = encode(model, queries)
faiss.normalize_L2(q_emb)
D,I = index.search(q_emb.astype('float32'), 10)

mrr=0; r1=r5=r10=0
for i,g in enumerate(gold_ids):
    ranks = I[i].tolist()
    if g in ranks:
        mrr += 1/(ranks.index(g)+1)
    if ranks[0]==g: r1+=1
    if g in ranks[:5]: r5+=1
    if g in ranks[:10]: r10+=1

n=len(queries)
print("\n==== FINAL AXL-ICT RESULTS ====")
print("MRR:", mrr/n)
print("Recall@1:", r1/n)
print("Recall@5:", r5/n)
print("Recall@10:", r10/n)


# save
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,"axl_ict_model.pt"))
faiss.write_index(index, os.path.join(OUTPUT_DIR,"faiss_index.idx"))
print("\nSaved AXL-ICT model + index to:", OUTPUT_DIR)
