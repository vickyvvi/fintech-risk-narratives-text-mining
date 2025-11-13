# === Quick Embedding A/B/C Report: FinBERT vs ProsusAI-FinBERT vs ModernBERT ===
# 依赖: pip install transformers torch pandas numpy scikit-learn matplotlib
import os, math, random, gc
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


DATA_PATH      = "/root/autodl-fs/sentences_trackB_A_ready_v73.csv"  # CSV 或 .parquet
TEXT_COL       = "sentence_raw"
NUM_SAMPLES    = 3000     # 2000~5000 足够轻量
MAX_LENGTH     = 256
BATCH_SIZE     = 64
SEED           = 42

MODELS = {
    "FinBERT-768":          "yiyanghkust/finbert-tone",
    "ProsusAI-FinBERT-768": "ProsusAI/finbert",
    "ModernBERT-1024":      "tyuan73/finetuned-modernbert-finance-large",
}

SAVE_PREFIX    = "/root/autodl-fs/ab_compare"  # 输出前缀

# --------- Utils ---------
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def load_texts(path, text_col, n):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    s = (df[text_col].astype(str)
         .str.strip()
         .replace("", np.nan)
         .dropna())
    s = s.drop_duplicates().sample(min(n, len(s)), random_state=SEED)
    s = s.tolist()
    print(f"[LOAD] texts={len(s)} from {os.path.basename(path)}")
    return s

@torch.no_grad()
def embed_texts(model_name, texts, max_length=256, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    dtype = torch.float16 if device=="cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(
            batch,
            padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(device)
        out = model(**enc)
        last = out.last_hidden_state  # [B, L, H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        summed = (last * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        emb = summed / counts
        emb = nn.functional.normalize(emb, p=2, dim=1)
        all_vecs.append(emb.detach().cpu())

    E = torch.cat(all_vecs, dim=0).numpy().astype("float32")  # 转为 float32，避免 NumPy 报错
    del model; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return E

def pairwise_cosine_stats(E, max_pairs=2_000_000):
    n = E.shape[0]
    if n*(n-1)//2 <= max_pairs:
        S = cosine_similarity(E)
        iu = np.triu_indices(n, k=1)
        vals = S[iu]
    else:
        idx1 = rng.integers(0, n, size=max_pairs)
        idx2 = rng.integers(0, n, size=max_pairs)
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        vals = np.sum(E[idx1]*E[idx2], axis=1)
    return vals

def knn_similarity(E, ks=(1,5,10)):
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    out = {}
    for k in ks:
        topk = np.partition(S, -k, axis=1)[:, -k:]
        out[f"top{k}_mean"] = float(topk.mean())
        out[f"top{k}_p90"]  = float(np.quantile(topk, 0.90))
    return out

def quick_silhouette(E, k=10):
    n, d = E.shape
    proj = rng.normal(size=(d, 8))
    P = E @ proj
    bins = np.floor((P - P.mean(0, keepdims=True)) / (P.std(0, keepdims=True)+1e-6)).astype(int)
    lab = np.mod(np.abs(bins.sum(1)), k)
    try:
        sil = silhouette_score(E, lab, metric="cosine")
    except Exception:
        sil = np.nan
    return float(sil)

def plot_hist(dists_dict, out_png):
    plt.figure(figsize=(7.5,5.2))
    for name, vals in dists_dict.items():
        plt.hist(vals, bins=60, alpha=0.5, label=name, density=True)
    plt.title("Pairwise cosine similarity distribution")
    plt.xlabel("cosine similarity"); plt.ylabel("density")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    print(f"[PLOT] {out_png}")

# --------- Run ---------
texts = load_texts(DATA_PATH, TEXT_COL, NUM_SAMPLES)

embeds = {}
for nice_name, hub_name in MODELS.items():
    E = embed_texts(hub_name, texts, MAX_LENGTH, BATCH_SIZE)
    embeds[nice_name] = E
    np.save(f"{SAVE_PREFIX}_{nice_name}.npy", E)
    print(f"[SAVE] {nice_name} -> {SAVE_PREFIX}_{nice_name}.npy  shape={E.shape}")

reports = []
pair_dists = {}

for name, E in embeds.items():
    norms = np.linalg.norm(E, axis=1)
    sv = np.linalg.svd(E - E.mean(0, keepdims=True), full_matrices=False, compute_uv=False)
    pair_vals = pairwise_cosine_stats(E)
    pair_dists[name] = pair_vals

    nn_stats = knn_similarity(E, ks=(1,5,10))
    sil = quick_silhouette(E, k=10)

    rep = {
        "model": name,
        "n_rows": E.shape[0],
        "dim": E.shape[1],
        "norm_mean": float(norms.mean()),
        "norm_std":  float(norms.std()),
        "sv1_frac":  float(sv[0] / (sv.sum()+1e-9)),
        "pair_mean": float(pair_vals.mean()),
        "pair_p90":  float(np.quantile(pair_vals, 0.90)),
        "pair_p99":  float(np.quantile(pair_vals, 0.99)),
        "frac_pair_gt_0.9": float((pair_vals > 0.9).mean()),
        "silhouette_approx_k10": sil,
        **nn_stats
    }
    reports.append(rep)

rep_df = pd.DataFrame(reports).sort_values("model")
out_csv = f"{SAVE_PREFIX}_summary.csv"
rep_df.to_csv(out_csv, index=False)
print("\n[SUMMARY]")
print(rep_df.to_string(index=False))
print(f"[SAVE] {out_csv}")

# 画配对相似度分布
out_png = f"{SAVE_PREFIX}_pairwise_hist.png"
plot_hist(pair_dists, out_png)

# Cross-model NN check: 只在两个模型时执行
if len(embeds) == 2:
    names = list(embeds.keys())
    A, B = embeds[names[0]], embeds[names[1]]
    SA = A @ A.T
    np.fill_diagonal(SA, -np.inf)
    nn_idx_A = SA.argmax(axis=1)
    sim_in_B = np.sum(B[np.arange(len(B))] * B[nn_idx_A], axis=1)
    print(f"\n[Cross-model NN check] {names[0]}->NN sim measured in {names[1]}:")
    print(f"  mean={sim_in_B.mean():.4f} | p90={np.quantile(sim_in_B,0.9):.4f} | p99={np.quantile(sim_in_B,0.99):.4f}")
