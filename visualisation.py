# =============================================================
#        VISUALISATION SUITE FOR ICT-P / AXL-ICT RETRIEVAL
# =============================================================

import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict

OUTPUT_DIR = "/content/ict_p"       # ← change if needed
PASSAGES_JSON = f"{OUTPUT_DIR}/passages.json"
PASS_EMB = f"{OUTPUT_DIR}/passages_emb.npy"
EVAL_JSON = f"{OUTPUT_DIR}/eval_metrics.json"
RETR_JSON = f"{OUTPUT_DIR}/sample_retrievals.json"

# Increase plot size globally
plt.rcParams['figure.figsize'] = [10, 7]
sns.set(style="whitegrid")

# -------------------------------------------------------------
# Load data
# -------------------------------------------------------------
print("Loading files...")

with open(PASSAGES_JSON, "r") as f:
    passages = json.load(f)

pass_emb = np.load(PASS_EMB)
with open(EVAL_JSON, "r") as f:
    metrics = json.load(f)

with open(RETR_JSON, "r") as f:
    retrievals = json.load(f)

print("Loaded:")
print(" - passages:", len(passages))
print(" - embeddings:", pass_emb.shape)
print(" - retrieval samples:", len(retrievals))

# -------------------------------------------------------------
# 1. CLUSTER VISUALISATION (PCA / TSNE)
# -------------------------------------------------------------
print("\nPlotting PCA (2D) of passage embeddings...")

pca = PCA(n_components=2)
p2 = pca.fit_transform(pass_emb)

plt.scatter(p2[:,0], p2[:,1], s=8, alpha=0.6)
plt.title("Passage Embedding Distribution (PCA-2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# Optional: t-SNE (slow)
print("Plotting t-SNE (this may take ~1–2 minutes for many passages)...")
tsne = TSNE(n_components=2, perplexity=40, n_iter=1500)
t2 = tsne.fit_transform(pass_emb[:3000])   # take first 3k for speed

plt.scatter(t2[:,0], t2[:,1], s=8, alpha=0.6)
plt.title("t-SNE Embedding Space (subset)")
plt.show()


# -------------------------------------------------------------
# 2. CLUSTER SIZE HISTOGRAM
# -------------------------------------------------------------
print("\nPlotting cluster size histogram...")
from sklearn.cluster import MiniBatchKMeans

k = max(2, len(pass_emb)//32)
km = MiniBatchKMeans(n_clusters=k).fit(pass_emb)
labels = km.labels_

sizes = defaultdict(int)
for l in labels:
    sizes[l]+=1

plt.hist(list(sizes.values()), bins=30, color="navy")
plt.title("Cluster Size Distribution")
plt.xlabel("Cluster Size")
plt.ylabel("Count")
plt.show()


# -------------------------------------------------------------
# 3. LOSS CURVE VISUALIZATION
# -------------------------------------------------------------
loss_log = f"{OUTPUT_DIR}/loss_log.json"
if os.path.exists(loss_log):
    with open(loss_log) as f:
        loss_data = json.load(f)

    epochs = list(range(1, len(loss_data)+1))
    losses = [x["loss"] for x in loss_data]

    plt.plot(epochs, losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
else:
    print("\n⚠ No loss_log.json found. (Add logging during training).")


# -------------------------------------------------------------
# 4. RETRIEVAL METRICS VISUALISATION
# -------------------------------------------------------------
print("\nPlotting retrieval metrics...")

bars = ["MRR", "Recall@1", "Recall@5", "Recall@10", "Recall@100"] if "Recall@100" in metrics else ["MRR","Recall@1","Recall@5","Recall@10"]
values = [metrics[k] for k in bars]

sns.barplot(x=bars, y=values, palette="viridis")
plt.ylim(0, 1)
plt.title("Retrieval Metrics")
plt.show()


# -------------------------------------------------------------
# 5. PER LANGUAGE METRICS (MRR@100 / Recall@100)
# -------------------------------------------------------------
langs = defaultdict(list)   # lang → retrieval objects

for r in retrievals:
    # auto-detect language from query text
    q = r["query"]
    if "\u0c00" <= q[0] <= "\u0c7f": lang = "te"
    elif "\u0600" <= q[0] <= "\u06ff": lang = "ar"
    elif "\u0e00" <= q[0] <= "\u0e7f": lang = "th"
    elif "\u0b80" <= q[0] <= "\u0bff": lang = "ta"
    elif "\u3040" <= q[0] <= "\u309f": lang = "jp"
    elif "\u4e00" <= q[0] <= "\u9fff": lang = "zh"
    else: lang = "en"

    langs[lang].append(r)

# compute per-language metrics
lang_mrr = {}
lang_r100 = {}

for lang, rows in langs.items():
    mrr = 0
    r100 = 0
    for r in rows:
        ranks = [x["passage_id"] for x in r["retrieved"]]
        gold = r["query_id"]
        if gold in ranks:
            mrr += 1/(ranks.index(gold)+1)
        if gold in ranks[:100]:
            r100 += 1
    n = len(rows)
    lang_mrr[lang] = mrr/n
    lang_r100[lang] = r100/n

# MRR per lang
plt.title("MRR@100 per Language")
sns.barplot(x=list(lang_mrr.keys()), y=list(lang_mrr.values()), palette="rocket")
plt.ylim(0,1)
plt.show()

# Recall@100 per lang
plt.title("Recall@100 per Language")
sns.barplot(x=list(lang_r100.keys()), y=list(lang_r100.values()), palette="mako")
plt.ylim(0,1)
plt.show()


# -------------------------------------------------------------
# 6. TOP-K SIMILARITY HEATMAP
# -------------------------------------------------------------
print("\nPlotting Top-K similarity heatmap (#50 samples)...")

sim_matrix = np.zeros((50,50))

for i in range(50):
    for j in range(50):
        # cheap similarity: dot product of embeddings
        sim_matrix[i,j] = np.dot(pass_emb[i], pass_emb[j])

sns.heatmap(sim_matrix, cmap="viridis")
plt.title("Similarity Heatmap (first 50 passages)")
plt.show()

print("\n✓ All visualizations completed.")
