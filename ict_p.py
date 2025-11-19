# DPR with ICT-P (paper-style)
# - Bi-encoder (mBERT) query+passage encoders
# - Pre-finetune on MS MARCO, then fine-tune on Mr.TyDi (or your combined set)
# - ICT-P: cluster passage representations before each epoch and build batches from clusters
# - In-batch contrastive (DPR negative log-likelihood) loss using dot product similarity

import os, math, random, time, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # optional to reduce fragmentation

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import faiss
import gc

# -------------------------
# Config (edit to your env)
# -------------------------
MSMARCO_CSV = "/content/msmarco_triples.csv"   # must have columns: query, positive, (optional negative)
TYDI_DIR    = "/content"                       # contains mrtydi_<lang>.json/.jsonl (optional)
OUTPUT_DIR  = "/content/ict_p"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Safe defaults (for local/Kaggle): you can increase if you have larger GPU
PRE_EPOCHS = 5      
FT_EPOCHS  = 5        
BATCH_SIZE = 4          
LR         = 1e-5
ACCUM_STEPS = 1         # gradient accumulation (increase if you need effective larger batch)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-multilingual-cased"  # paper uses mBERT (bert-base-multilingual-cased)
MAX_LEN = 64           # token length for queries/passages (increase if passages are longer)
POOLING = "mean"       # pooling strategy for sentence embedding
CLUSTERING_K_FACTOR = 1.0  # k = ceil(num_passages / (BATCH_SIZE*CLUSTERING_K_FACTOR)) adjust for cluster size
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# Helpers: dataset & tokenize
# -------------------------
class QPDataset(Dataset):
    def __init__(self, rows):
        # rows is a list of dicts: {"q":..., "pos":...}
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        return r["q"], r["pos"], idx

def collate_texts_texts(batch, tokenizer, max_len=MAX_LEN):
    # batch: list of (q, pos, idx)
    queries = [b[0] for b in batch]
    passages = [b[1] for b in batch]
    q_tokens = tokenizer(queries, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    p_tokens = tokenizer(passages, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return q_tokens, p_tokens

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]   # (batch, seq_len, dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# -------------------------
# DPR model wrapper
# -------------------------
class DPR_BiEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Two independent mBERT models (query and passage encoders)
        self.q_encoder = AutoModel.from_pretrained(model_name)
        self.p_encoder = AutoModel.from_pretrained(model_name)
        # optional projection (paper uses raw BERT outputs -> dot product)
        # you may add a projection to reduce dim if wanted
        # self.proj = nn.Linear(self.q_encoder.config.hidden_size, 768)
    def forward_query(self, input_ids, attention_mask):
        out = self.q_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        emb = mean_pooling(out, attention_mask)
        return emb
    def forward_passage(self, input_ids, attention_mask):
        out = self.p_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        emb = mean_pooling(out, attention_mask)
        return emb
    def to_device(self, device):
        self.to(device)

# -------------------------
# Utility: encode texts in batches (CPU/GPU safe)
# -------------------------
def encode_texts_encoder(encoder_model, tokenizer, texts, device, batch_size=64, max_len=MAX_LEN, is_query=False):
    encoder_model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            toks = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            input_ids = toks["input_ids"].to(device)
            attn = toks["attention_mask"].to(device)
            if is_query:
                out = encoder_model.forward_query(input_ids, attn)
            else:
                out = encoder_model.forward_passage(input_ids, attn)
            all_embs.append(out.cpu().numpy())
    return np.vstack(all_embs)

# -------------------------
# Loss: in-batch DPR NLL
# -------------------------
def inbatch_dpr_loss(q_emb, p_emb, temperature=1.0):
    # q_emb: (B, D) ; p_emb: (B, D)
    # compute similarity matrix
    logits = torch.matmul(q_emb, p_emb.t()) / temperature
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    # compute simple metrics (recall@1)
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        r1 = (preds == labels).float().mean().item()
    return loss, r1

# -------------------------
# ICT-P training epoch (cluster passages, build batches)
# -------------------------
def build_ictp_batches(passage_embeddings, batch_size, k_factor=CLUSTERING_K_FACTOR, seed=SEED):
    # passage_embeddings: numpy (N, D)
    N = passage_embeddings.shape[0]
    # number of clusters
    # Paper: cluster into k clusters -> then sub-split into batches of size b
    # simple heuristic: k = ceil(N / (b * k_factor))
    k = max(1, int(math.ceil(N / (batch_size * k_factor))))
    # MiniBatchKMeans for speed
    mbk = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1024)
    mbk.fit(passage_embeddings)
    labels = mbk.labels_
    # group indices by cluster
    clusters = {}
    for idx, c in enumerate(labels):
        clusters.setdefault(c, []).append(idx)
    # for each cluster, split into chunks of size batch_size
    batches = []
    for c_idx, idxs in clusters.items():
        # shuffle indices within cluster to add randomness
        random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            chunk = idxs[i:i+batch_size]
            if len(chunk) < batch_size:
                # we can either drop small tail or pad by sampling random others from same cluster
                # Paper combined small sub-clusters until size b (they combine small subclusters). We approximate:
                # Skip tiny tails to keep balanced batches (or pad by random choices to full size)
                # Here we will skip tails smaller than batch_size // 2 to avoid tiny batches
                if len(chunk) < max(1, batch_size//2):
                    continue
                # else include the smaller tail as a batch
            batches.append(chunk)
    # shuffle batches globally
    random.shuffle(batches)
    return batches

# -------------------------
# Training loops
# -------------------------
def train_stage(model, tokenizer, train_rows, epochs, batch_size, lr, device, stage_name="FT", clustering=True, update_every_epoch=True, save_prefix="model_stage"):
    """
    train_rows: list of {"q":..., "pos":...}
    """
    model.to(device)
    optimizer = AdamW(list(model.q_encoder.parameters()) + list(model.p_encoder.parameters()), lr=lr)
    total_steps = max(1, (len(train_rows) // batch_size) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= max(1, int(0.01*total_steps)), num_training_steps=total_steps)
    # For gradient accumulation
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))
    global_step = 0

    # Prepare tokenizer shortcuts
    def tokenize_qs(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    def tokenize_ps(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

    for epoch in range(1, epochs+1):
        print(f"\n=== {stage_name} epoch {epoch}/{epochs} ===")
        model.train()

        # ICT-P clustering: compute passage embeddings and cluster
        if clustering:
            print("Encoding all passages for clustering (batch)...")
            passages = [r["pos"] for r in train_rows]
            # encode using passage encoder
            p_embs = encode_texts_encoder(model, tokenizer, passages, device, batch_size=128, max_len=MAX_LEN, is_query=False)
            print("Passage embeddings shape:", p_embs.shape)
            # build batches from clusters
            batches = build_ictp_batches(p_embs, batch_size)
            print("Number of batches (clustered):", len(batches))
        else:
            # simple sequential batching
            N = len(train_rows)
            indices = list(range(N))
            random.shuffle(indices)
            batches = [indices[i:i+batch_size] for i in range(0, N, batch_size)]
            print("Number of batches (sequential):", len(batches))

        epoch_loss = 0.0
        epoch_r1 = 0.0
        pbar = tqdm(batches, desc=f"{stage_name}-epoch{epoch}")
        optimizer.zero_grad()
        for step, batch_idxs in enumerate(pbar):
            # build lists of queries & passages from indices
            qs = [train_rows[i]["q"] for i in batch_idxs]
            ps = [train_rows[i]["pos"] for i in batch_idxs]
            # tokenize
            q_toks = tokenizer(qs, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            p_toks = tokenizer(ps, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

            q_input_ids = q_toks["input_ids"].to(device)
            q_attn = q_toks["attention_mask"].to(device)
            p_input_ids = p_toks["input_ids"].to(device)
            p_attn = p_toks["attention_mask"].to(device)

            # forward with mixed-precision if GPU
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                q_emb = model.forward_query(q_input_ids, q_attn)   # (B, D)
                p_emb = model.forward_passage(p_input_ids, p_attn) # (B, D)
                loss, r1 = inbatch_dpr_loss(q_emb, p_emb, temperature=1.0)

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)  # optional
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item()
            epoch_r1 += r1
            pbar.set_postfix(loss=epoch_loss/(step+1), r1=epoch_r1/(step+1))

            # free mem each step
            del q_input_ids, q_attn, p_input_ids, p_attn, q_emb, p_emb, loss
            torch.cuda.empty_cache()

        # epoch summary
        avg_loss = epoch_loss / max(1, len(batches))
        avg_r1 = epoch_r1 / max(1, len(batches))
        print(f"Epoch {epoch} summary - loss: {avg_loss:.4f}  Recall@1 (batch): {avg_r1:.4f}")

        # optionally update representations less frequently (paper updated every 10 epochs for iterative methods)
        # here we recompute passages at every epoch because we cluster every epoch (ICT-P)
        # Save model checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f"{save_prefix}_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": {"q_encoder": model.q_encoder.state_dict(), "p_encoder": model.p_encoder.state_dict()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

    print("Training stage done.")
    return model

# -------------------------
# Load MS MARCO + MrTyDi (simple robust loaders)
# -------------------------
def load_msm_csv(path, sample_limit=1000, seed=SEED):
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        quoting=3
    )
    cols = {c.lower():c for c in df.columns}
    if "query" not in cols or "positive" not in cols:
        raise ValueError("MSMARCO CSV must have columns 'query' and 'positive' (case-insensitive).")
    qcol = cols["query"]; pcol = cols["positive"]; ncol = cols.get("negative", None)
    if sample_limit:
        df = df.sample(sample_limit, random_state=seed).reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        rows.append({"q": str(r[qcol]), "pos": str(r[pcol])})
    return rows

def load_mrtydi_from_dir(tydi_dir, per_lang_limit=100):
    # robustly load mrtydi_*.json(.jsonl)
    rows = []
    for fn in sorted(os.listdir(tydi_dir)):
        if fn.lower().startswith("mrtydi_") and (fn.lower().endswith(".json") or fn.lower().endswith(".jsonl")):
            path = os.path.join(tydi_dir, fn)
            try:
                with open(path, "r", encoding="utf8") as f:
                    first = f.read(1)
                    f.seek(0)
                    if first.strip().startswith("["):
                        arr = json.load(f)
                        for ob in arr[:per_lang_limit] if per_lang_limit else arr:
                            q = ob.get("query") or ob.get("question") or ob.get("query_text") or ""
                            p = ob.get("positive_passage") or ob.get("positive") or ob.get("passage") or ob.get("text") or ""
                            if q and p:
                                rows.append({"q": str(q), "pos": str(p)})
                    else:
                        for i,line in enumerate(f):
                            if per_lang_limit and i>=per_lang_limit: break
                            if not line.strip(): continue
                            try:
                                ob = json.loads(line)
                            except:
                                continue
                            q = ob.get("query") or ob.get("question") or ob.get("query_text") or ""
                            p = ob.get("positive_passage") or ob.get("positive") or ob.get("passage") or ob.get("text") or ""
                            if q and p:
                                rows.append({"q": str(q), "pos": str(p)})
            except Exception as e:
                print("Failed reading", path, e)
    return rows

# -------------------------
# Main: build, pre-train, finetune, eval & index
# -------------------------
def main():
    # load data
    print("Loading MS MARCO (pre-finetune)...")
    msm_rows = load_msm_csv(MSMARCO_CSV, sample_limit=1000)   # PAPER uses full MS MARCO
    print("MSMARCO rows:", len(msm_rows))
    print("Loading Mr.TyDi (fine-tune) from directory (if available)...")
    tydi_rows = load_mrtydi_from_dir(TYDI_DIR, per_lang_limit=100)
    print("Mr.TyDi rows:", len(tydi_rows))

    # If no tydi files, just fine-tune on MS MARCO (or user data)
    ft_rows = tydi_rows if len(tydi_rows) > 0 else msm_rows

    # build model & tokenizer
    print("Building DPR bi-encoder model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dpr = DPR_BiEncoder(MODEL_NAME)

    # Optional memory improvements
    try:
        dpr.q_encoder.gradient_checkpointing_enable()
        dpr.p_encoder.gradient_checkpointing_enable()
    except Exception:
        pass

    # Pre-finetune on MS MARCO (paper: 40 epochs)
    if PRE_EPOCHS > 0 and len(msm_rows) > 0:
        print("=== PRE-FINETUNE (MSMARCO) ===")
        dpr = train_stage(dpr, tokenizer, msm_rows, epochs=PRE_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                          device=DEVICE, stage_name="PREFT", clustering=False, save_prefix="prefinetune")
    else:
        print("Skipping pre-finetune (PRE_EPOCHS==0)")

    # Fine-tune on Mr.TyDi (paper: 40 epochs), ICT-P clustering enabled
    if FT_EPOCHS > 0 and len(ft_rows) > 0:
        print("=== FINETUNE (Mr.TyDi / FT rows) ===")
        dpr = train_stage(dpr, tokenizer, ft_rows, epochs=FT_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                          device=DEVICE, stage_name="FT", clustering=True, save_prefix="finetune")
    else:
        print("Skipping fine-tune (FT_EPOCHS==0 or no FT rows)")

    # Build FAISS index from all passages (we use passages from ft_rows here)
    print("Building corpus embeddings and FAISS index (for evaluation)...")
    passages = [r["pos"] for r in ft_rows]
    queries = [r["q"] for r in ft_rows]
    # encode passages
    p_embs = encode_texts_encoder(dpr, tokenizer, passages, DEVICE, batch_size=128, max_len=MAX_LEN, is_query=False)
    import faiss
    faiss.normalize_L2(p_embs)
    d = p_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(p_embs.astype('float32'))
    print("FAISS index built, npassages:", index.ntotal)
    # Save index (CPU version)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss_index.idx"))
    np.save(os.path.join(OUTPUT_DIR, "passages_emb.npy"), p_embs)
    with open(os.path.join(OUTPUT_DIR, "passages.json"), "w", encoding="utf8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    print("Saved corpus and index to", OUTPUT_DIR)

    print("Done. You can now run evaluation using the FAISS index and query encoder.")

if __name__ == "__main__":
    main()
