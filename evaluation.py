# ====== DPR evaluation + sample retrievals ======

import os, json, math, time
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import faiss
from torch import nn

# ---------- USER PATHS ----------
OUTPUT_DIR = "/content/ict_p"           # where checkpoints, index, passages.json are saved
CKPT_PREF = os.path.join(OUTPUT_DIR, "finetune_epoch1.pt")   # or prefintune... choose appropriate
FAISS_IDX = os.path.join(OUTPUT_DIR, "faiss_index.idx")
PASSAGES_JSON = os.path.join(OUTPUT_DIR, "passages.json")
PASS_EMB_NPY = os.path.join(OUTPUT_DIR, "passages_emb.npy")  # optional
# ---------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-multilingual-cased"   # same used before
MAX_LEN = 64
BATCH = 64   # batch size for encoding queries/passages at eval

print("DEVICE:", DEVICE)
print("Loading tokenizer & building model wrapper...")

# ---- model wrapper (must match training wrapper) ----
class DPR_BiEncoderEval(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.q_encoder = AutoModel.from_pretrained(model_name)
        self.p_encoder = AutoModel.from_pretrained(model_name)
    def forward_query(self, input_ids, attention_mask):
        out = self.q_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        token_embeddings = out[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        return summed / summed_mask
    def forward_passage(self, input_ids, attention_mask):
        out = self.p_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        token_embeddings = out[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        return summed / summed_mask

# load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DPR_BiEncoderEval(MODEL_NAME).to(DEVICE)
model.eval()

# If you saved checkpoint state dicts, load them (works with the saving format used earlier)
if os.path.exists(CKPT_PREF):
    ck = torch.load(CKPT_PREF, map_location="cpu")
    q_state = ck.get("q_state") or (ck.get("model_state") and ck["model_state"].get("q_encoder"))
    p_state = ck.get("p_state") or (ck.get("model_state") and ck["model_state"].get("p_encoder"))
    if q_state and p_state:
        model.q_encoder.load_state_dict(q_state)
        model.p_encoder.load_state_dict(p_state)
        print("Loaded checkpoint states from", CKPT_PREF)
    else:
        print("Checkpoint exists but state keys not found; using fresh model weights.")
else:
    print("Checkpoint not found at", CKPT_PREF, " — using model weights from huggingface")

# ----- helpers -----
def mean_pooling_from_model_output(out, attention_mask):
    token_embeddings = out[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    return (summed / summed_mask)

@torch.no_grad()
def encode_texts(encoder, texts, tokenizer, device, batch_size=64, max_len=MAX_LEN, is_query=True):
    encoder.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = toks["input_ids"].to(device)
        attn = toks["attention_mask"].to(device)
        if is_query:
            out = encoder.forward_query(input_ids, attn)
        else:
            out = encoder.forward_passage(input_ids, attn)
        embs.append(out.cpu().numpy())
        del input_ids, attn, out
        torch.cuda.empty_cache()
    if len(embs)==0:
        return np.zeros((0, encoder.q_encoder.config.hidden_size))
    return np.vstack(embs)

# ----- load passages and queries -----
if os.path.exists(PASSAGES_JSON):
    with open(PASSAGES_JSON, "r", encoding="utf8") as f:
        passages = json.load(f)
else:
    raise FileNotFoundError("Passages file not found at: " + PASSAGES_JSON)

# We'll treat queries = passages' queries if you saved them earlier; otherwise run queries from MSMARCO file again.
# For evaluation here we assume queries are the same length as passages (gold is i->i)
queries = []
# if you stored queries in 'passages.json' as pairs, adapt accordingly. For now, use the same number and dummy queries if necessary:
# try to load a queries.json if exists
QPATH = os.path.join(OUTPUT_DIR, "queries.json")
if os.path.exists(QPATH):
    with open(QPATH, "r", encoding="utf8") as f:
        queries = json.load(f)
else:
    # fallback: construct dummy queries by re-loading msmarco or tydi sources if you have them saved
    # If you don't have queries, we'll assume gold_ids = range(npassages) and use queries = passages (not ideal)
    print("No queries.json found — using passages as queries fallback (gold is index->index).")
    queries = passages.copy()

n = len(queries)
print("n_queries:", n, "n_passages:", len(passages))

# ----- load faiss index -----
if os.path.exists(FAISS_IDX):
    index = faiss.read_index(FAISS_IDX)
    print("Loaded FAISS index from", FAISS_IDX, "npassages:", index.ntotal)
else:
    # if not saved, build from scratch using model to encode passages (slow)
    print("FAISS index not found, encoding passages now and building index...")
    p_embs = encode_texts(model, passages, tokenizer, DEVICE, batch_size=BATCH, max_len=MAX_LEN, is_query=False)
    faiss.normalize_L2(p_embs)
    dim = p_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(p_embs.astype("float32"))
    faiss.write_index(index, FAISS_IDX)
    print("Built and saved FAISS index.")

# ----- Evaluate: compute query embs and search -----
TOP_K = 10
batch = BATCH
print("Encoding queries and searching (top k):", TOP_K)
mrr_total = 0.0
r1 = r5 = r10 = 0
all_retrievals = []

for i in range(0, n, batch):
    batch_q = queries[i:i+batch]
    q_emb = encode_texts(model, batch_q, tokenizer, DEVICE, batch_size=batch, max_len=MAX_LEN, is_query=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), TOP_K)   # D: scores, I: indices
    for j in range(len(batch_q)):
        qidx = i + j
        gold = qidx   # assuming gold id = index (if you used same order during train)
        ranks = I[j].tolist()
        # MRR
        if gold in ranks:
            pos = ranks.index(gold) + 1
            mrr_total += 1.0 / pos
        # recalls
        if ranks[0] == gold:
            r1 += 1
        if gold in ranks[:5]:
            r5 += 1
        if gold in ranks[:10]:
            r10 += 1
        # store sample retrievals for first 50 queries
        if qidx < 50:
            retrieved = []
            for rr, idx in enumerate(ranks):
                retrieved.append({"rank": rr+1, "passage_id": int(idx), "passage": passages[idx], "score": float(D[j][rr])})
            all_retrievals.append({"query_id": qidx, "query": batch_q[j], "gold_id": gold, "retrieved": retrieved})

# final metrics
nq = n if n>0 else 1
metrics = {
    "MRR": mrr_total / nq,
    "Recall@1": r1 / nq,
    "Recall@5": r5 / nq,
    "Recall@10": r10 / nq,
    "n_queries": nq
}
print("=== EVAL METRICS ===")
for k,v in metrics.items():
    print(f"{k}: {v:.4f}")

# save outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w", encoding="utf8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTPUT_DIR, "sample_retrievals.json"), "w", encoding="utf8") as f:
    json.dump(all_retrievals, f, ensure_ascii=False, indent=2)

print("Saved eval_metrics.json and sample_retrievals.json to", OUTPUT_DIR)
