# ============================================================
#            PER-LANGUAGE EVALUATION (MRR@10 / MRR@100)
# ============================================================

print("\n=== Running Per-Language Evaluation ===")

from collections import defaultdict

# Prepare query texts + assign language using language-map
query_texts = []
query_langs = []
query_gold = []

for i, q in enumerate(queries):

    # text extraction
    if isinstance(q, str):
        text = q
    elif isinstance(q, dict):
        text = q.get("query") or q.get("text") or q.get("q") or next(iter(q.values()))
    else:
        text = str(q)

    text_clean = text.strip()

    # language lookup from map
    lang = tydi_map.get(text_clean, "unknown")
    query_langs.append(lang)

    # gold id: identity mapping
    query_texts.append(text_clean)
    query_gold.append(i)

# Storage for metrics
perlang = defaultdict(lambda: {
    "count":0,
    "mrr10":0.0,
    "mrr100":0.0,
    "r1":0,
    "r5":0,
    "r10":0,
    "r100":0
})

TOP_K = 100
print("TOP_K =", TOP_K)

for i in range(0, len(query_texts), BATCH):

    batch_q = query_texts[i:i+BATCH]
    q_emb = encode_texts(model, batch_q, tokenizer, DEVICE, batch_size=BATCH, max_len=MAX_LEN, is_query=True)

    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), TOP_K)

    for j in range(len(batch_q)):
        qidx = i + j
        gold = query_gold[qidx]
        lang = query_langs[qidx]
        ranks = I[j].tolist()

        perlang[lang]["count"] += 1

        # MRR@10
        if gold in ranks[:10]:
            pos10 = ranks[:10].index(gold) + 1
            perlang[lang]["mrr10"] += 1.0 / pos10

        # MRR@100
        if gold in ranks:
            pos100 = ranks.index(gold) + 1
            perlang[lang]["mrr100"] += 1.0 / pos100

        # Recalls
        if ranks[0] == gold:
            perlang[lang]["r1"] += 1
        if gold in ranks[:5]:
            perlang[lang]["r5"] += 1
        if gold in ranks[:10]:
            perlang[lang]["r10"] += 1
        if gold in ranks[:100]:
            perlang[lang]["r100"] += 1


# --------- Aggregate Final Language Metrics ----------
per_language_metrics = {}

for lang, s in perlang.items():
    c = s["count"]
    per_language_metrics[lang] = {
        "MRR@10": s["mrr10"] / c if c else 0,
        "MRR@100": s["mrr100"] / c if c else 0,
        "Recall@10": s["r10"] / c if c else 0,
        "Recall@100": s["r100"] / c if c else 0,
    }

print("\n========== PER-LANGUAGE RESULTS ==========\n")
for lang, m in per_language_metrics.items():
    print(f"Language = {lang}")
    for k,v in m.items():
        print(f"   {k}: {v:.4f}")
    print()

with open(os.path.join(OUTPUT_DIR, "per_language_metrics.json"), "w", encoding="utf8") as f:
    json.dump(per_language_metrics, f, ensure_ascii=False, indent=2)

print("\nSaved per_language_metrics.json")
