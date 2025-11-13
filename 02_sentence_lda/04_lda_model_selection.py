from pathlib import Path
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import tomotopy as tp
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

# === 路径 ===
DEDUP_PATH = Path("/root/autodl-fs/sentlda_final_v2_noempty.parquet")
MODEL_DIR  = Path("/root/autodl-fs/lda_sentence_grid"); MODEL_DIR.mkdir(exist_ok=True, parents=True)

# === 列名 ===
TEXT_COL = "tokens_final"   # 列里应是 list[str] 或可解析的字符串
ID_COL   = "sent_id"

# ---------- 工具：把任意形态的 token 列变成 List[str] ----------
def to_list_tokens(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, np.ndarray):
        return [str(t).strip() for t in x.ravel().tolist() if str(t).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # 形如 "['a','b']" / "('a','b')" -> 解析
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple, np.ndarray)):
                    seq = val.tolist() if isinstance(val, np.ndarray) else val
                    return [str(t).strip() for t in seq if str(t).strip()]
            except Exception:
                pass
        # 退化为空格分词（若是 "a b c"）
        return [w for w in s.split() if w.strip()]
    # 其他可迭代
    try:
        return [str(t).strip() for t in list(x) if str(t).strip()]
    except Exception:
        return []

# 读数据
df = pd.read_parquet(DEDUP_PATH).dropna(subset=[TEXT_COL]).reset_index(drop=True)

# 规范化 tokens（不要 .astype(str).split()）
tokens = [to_list_tokens(x) for x in tqdm(df[TEXT_COL].tolist(), desc="Normalize tokens")]

# 构建固定词典
dictionary = Dictionary(tokens)
dictionary.filter_extremes(no_below=20, no_above=0.4, keep_n=50000)
dictionary.compactify()

# 仅保留词典内 token
token_set = set(dictionary.values())
tokens_filtered = [[w for w in doc if w in token_set] for doc in tokens]

# 过滤空文档
mask = [len(doc) > 0 for doc in tokens_filtered]
tokens_filtered = [doc for doc, keep in zip(tokens_filtered, mask) if keep]
df_used = df.loc[mask].reset_index(drop=True)

# 保存词典与 tokens（用 parquet/pickle，避免 np.save 的形状问题）
dictionary.save("/root/autodl-fs/lda_dict.dict")
pd.DataFrame({"tokens": tokens_filtered}).to_parquet("/root/autodl-fs/tokens_filtered.parquet", index=False)
import pickle
with open("/root/autodl-fs/tokens_filtered.pkl", "wb") as f:
    pickle.dump(tokens_filtered, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[ok] docs used: {len(tokens_filtered)}, vocab: {len(dictionary)}")

def train_lda_and_metrics(K, tokens_docs, dictionary, iters=800, seed=42, topn=15):
    """
    训练一个 K 的 LDA 并返回：
      - 模型对象
      - perplexity（越低越好）
      - coherence_c_v（越高越好）
    """
    lda = tp.LDAModel(k=K, tw=tp.TermWeight.IDF, seed=seed)
    for doc in tokens_docs:
        lda.add_doc(doc)

    # 迭代训练（分批以避免长时间无输出）
    step = 50
    rounds = max(1, iters // step)
    for _ in range(rounds):
        lda.train(step)

    # Perplexity
    ll_per_word = getattr(lda, "ll_per_word", None)
    perp = float(np.exp(-ll_per_word)) if ll_per_word is not None else np.nan

    # Coherence (c_v)
    topics_top = [[w for w, _ in lda.get_topic_words(k, top_n=topn)] for k in range(K)]
    cm = CoherenceModel(topics=topics_top, texts=tokens_docs, dictionary=dictionary, coherence='c_v')
    coh_cv = float(cm.get_coherence())

    return lda, perp, coh_cv

K_list = list(range(5, 100, 1))
results = []

for K in K_list:
    print(f"\n=== Training K={K} ===")
    lda, perp, coh_cv = train_lda_and_metrics(
        K,
        tokens_filtered,
        dictionary,
        iters=800,
        seed=42,
        topn=15
    )

    # 存模型与主题词
    lda.save((MODEL_DIR / f"lda_K{K}.bin").as_posix())
    with open(MODEL_DIR / f"topics_K{K}.txt", "w", encoding="utf-8") as f:
        for k in range(K):
            words = [w for w, _ in lda.get_topic_words(k, top_n=15)]
            f.write(f"Topic {k:02d}: {' '.join(words)}\n")

    results.append({"K": K, "perplexity": perp, "coherence_c_v": coh_cv})

# 汇总表
res_df = pd.DataFrame(results)
res_df.to_csv(MODEL_DIR / "grid_results.csv", index=False)
print(res_df)

# 打分（coherence 越大越好；perplexity 越小越好 -> 取倒数归一）
dfm = res_df.copy()
dfm["perp_inv"] = 1.0 / (dfm["perplexity"].replace(0, np.nan) + 1e-9)
for col in ["perp_inv", "coherence_c_v"]:
    mn, mx = dfm[col].min(), dfm[col].max()
    dfm[col + "_norm"] = (dfm[col] - mn) / (mx - mn + 1e-9)
dfm["combo_score"] = 0.5 * dfm["perp_inv_norm"] + 0.5 * dfm["coherence_c_v_norm"]
dfm = dfm.sort_values("combo_score", ascending=False)
dfm.to_csv(MODEL_DIR / "grid_results_scored.csv", index=False)
print("\nTop by combo_score:")
print(dfm.head(5))

# 绘图
df_plot = res_df.sort_values("K").reset_index(drop=True)

plt.figure(figsize=(7, 4.5))
plt.plot(df_plot["K"], df_plot["perplexity"], marker="o")
plt.xlabel("Number of Topics (K)")
plt.ylabel("Perplexity (lower is better)")
plt.title("Perplexity vs. K")
idx_min_perp = df_plot["perplexity"].idxmin()
best_K_perp  = int(df_plot.loc[idx_min_perp, "K"])
best_perp    = float(df_plot.loc[idx_min_perp, "perplexity"])
plt.scatter([best_K_perp], [best_perp])
plt.annotate(f"best K={best_K_perp}\nperp={best_perp:.2f}",
             (best_K_perp, best_perp), xytext=(5,10), textcoords="offset points")
out1 = MODEL_DIR / "perplexity_vs_K.png"
plt.tight_layout(); plt.savefig(out1, dpi=180); plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(df_plot["K"], df_plot["coherence_c_v"], marker="o")
plt.xlabel("Number of Topics (K)")
plt.ylabel("Coherence c_v (higher is better)")
plt.title("Coherence (c_v) vs. K")
idx_max_coh = df_plot["coherence_c_v"].idxmax()
best_K_coh  = int(df_plot.loc[idx_max_coh, "K"])
best_coh    = float(df_plot.loc[idx_max_coh, "coherence_c_v"])
plt.scatter([best_K_coh], [best_coh])
plt.annotate(f"best K={best_K_coh}\nc_v={best_coh:.3f}",
             (best_K_coh, best_coh), xytext=(5,10), textcoords="offset points")
out2 = MODEL_DIR / "coherence_cv_vs_K.png"
plt.tight_layout(); plt.savefig(out2, dpi=180); plt.close()

print(f"已保存：\n- {out1}\n- {out2}")

