# -*- coding: utf-8 -*-
"""
Export per-cluster: TF-IDF keywords + representative sentences.
- 自动检测最终簇列（merged_v2 → merged → cluster_p3）
- 有嵌入 emb_* 时：代表句用嵌入簇心余弦；否则 TF-IDF 簇质心点积
- 导出：
  cluster_keywords.csv
  cluster_representative_sentences.csv
  cluster_naming_sheet.csv（每簇一行：簇ID、规模、前若干关键词、Top1代表句）
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ========== 配置 ==========
DATA_PATHS = [
    "sentences_with_cluster_p3_merged_v2.parquet",  # 优先使用第二次合并后的数据
    "sentences_with_cluster_p3_merged.parquet",     # 回退：第一次合并
    "/content/sentences_with_sentiment_plus_mpnet_no384_umap128_gpu_hdbscan_p3.parquet",  # 原始（不推荐）
]
TEXT_COL = "sentence_raw"
CAND_CLUSTER_COLS = ["cluster_p3_merged_v2", "cluster_p3_merged", "cluster_p3"]
EMB_PREFIX = "emb_"

TOP_K_KEYWORDS = 20
REP_PER_CLUSTER = 5

OUTDIR = Path("cluster_merge_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
# ==========================

# 1) 读取数据 & 选择簇列
df = None
for p in DATA_PATHS:
    if Path(p).exists():
        df = pd.read_parquet(p, engine="pyarrow"); DATA_PATH = p
        break
if df is None:
    raise FileNotFoundError("找不到数据文件，请把 DATA_PATHS 里至少一个路径改成你的实际文件。")

cluster_col = None
for c in CAND_CLUSTER_COLS:
    if c in df.columns:
        cluster_col = c; break
if cluster_col is None:
    raise KeyError(f"未检测到簇列，候选列为：{CAND_CLUSTER_COLS}")

assert TEXT_COL in df.columns, f"缺少文本列 {TEXT_COL}"

# 统一为字符串簇ID并去噪声
df[cluster_col] = df[cluster_col].astype(str)
df = df[df[cluster_col] != "-1"].copy()

# 2) 规模统计（方便排序与命名辅助）
size_tbl = (
    df.groupby(cluster_col).size()
      .reset_index(name="n_sentences")
      .sort_values("n_sentences", ascending=False)
      .reset_index(drop=True)
)
clusters = size_tbl[cluster_col].tolist()
print(f"[INFO] 数据源: {DATA_PATH}，簇数（去噪声）= {len(clusters)}")

# 3) 以簇为单位拼接文档，做 TF-IDF（关键词和 TF-IDF 方案都要用）
cluster_docs = (
    df.groupby(cluster_col)[TEXT_COL]
      .apply(lambda s: "\n".join(map(str, s)))
      .reindex(clusters)
)

# 中文/金融文本友好：使用 (1,2) ngram；可按需增大 max_features
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
X_tfidf = tfidf.fit_transform(cluster_docs.values)   # (n_clusters, vocab)
tfidf_vectors = normalize(X_tfidf, norm="l2", axis=1)
vocab = np.array(tfidf.get_feature_names_out())

# 4) 提取每簇 Top-K 关键词
kw_rows = []
for i, cid in enumerate(clusters):
    row = X_tfidf.getrow(i)
    if row.nnz == 0:
        kw_rows.append({"cluster": cid, "top_keywords": ""})
        continue
    dense = row.toarray().ravel()  # 稀疏转稠密再扁平
    top_idx = dense.argsort()[::-1][:TOP_K_KEYWORDS]
    words = vocab[top_idx]
    kw_rows.append({"cluster": cid, "top_keywords": ", ".join(words.tolist())})

kw_df = pd.DataFrame(kw_rows).merge(size_tbl, left_on="cluster", right_on=cluster_col, how="left")
kw_df = kw_df[["cluster", "n_sentences", "top_keywords"]]
kw_df.to_csv(OUTDIR/"cluster_keywords.csv", index=False)
print("[INFO] 已导出：cluster_keywords.csv")

# 5) 代表句：优先嵌入，否则 TF-IDF 点积
emb_cols = [c for c in df.columns if c.startswith(EMB_PREFIX)]
HAS_EMB = len(emb_cols) > 0
rep_rows = []

if HAS_EMB:
    # 用嵌入簇心余弦
    print(f"[INFO] 检测到嵌入列 {len(emb_cols)} 个，用嵌入挑代表句（簇心余弦）。")
    emb = df[emb_cols].to_numpy(dtype=np.float32)  # (n_sentences, dim)
    # 预先做向量归一化，加速相似度
    emb = normalize(emb, norm="l2", axis=1)
    # 为每个簇计算簇心（未归一化的均值后再归一）
    centroids = []
    for cid in clusters:
        mask = (df[cluster_col].values == cid)
        if mask.sum() == 0:
            centroids.append(np.zeros((emb.shape[1],), dtype=np.float32))
        else:
            c = emb[mask].mean(axis=0)
            nrm = np.linalg.norm(c) + 1e-12
            centroids.append(c / nrm)
    centroids = np.vstack(centroids)  # (n_clusters, dim)
    cid_to_cent = {cid: centroids[i] for i, cid in enumerate(clusters)}

    # 分簇挑代表句（Top-N 相似度）
    for cid in clusters:
        mask = (df[cluster_col].values == cid)
        idxs = np.where(mask)[0]
        if idxs.size == 0: 
            continue
        vecs = emb[idxs]                           # (m, dim), 已单位化
        c = cid_to_cent[cid].reshape(1, -1)        # (1, dim)
        sims = (vecs @ c.T).ravel()                # (m,)
        top_loc = np.argsort(-sims)[:min(REP_PER_CLUSTER, idxs.size)]
        for rank, loc in enumerate(top_loc, 1):
            sent_idx = idxs[loc]
            rep_rows.append({
                "cluster": cid,
                "rank": rank,
                "score_sim": float(sims[loc]),
                "sentence": df.iloc[sent_idx][TEXT_COL]
            })
else:
    # 用 TF-IDF：句子向量与簇 TF-IDF 质心的点积
    print("[INFO] 未检测到嵌入列，用 TF-IDF 点积挑代表句。")
    X_sent = tfidf.transform(df[TEXT_COL].astype(str).values)  # (n_sentences, vocab)
    for i, cid in enumerate(clusters):
        idxs = np.where(df[cluster_col].values == cid)[0]
        if idxs.size == 0:
            continue
        v_c = tfidf_vectors[i].T  # (vocab, 1)
        scores = X_sent[idxs].dot(v_c).toarray().ravel()  # 稀疏×稀疏→稀疏，转稠密再扁平
        top_loc = np.argsort(-scores)[:min(REP_PER_CLUSTER, idxs.size)]
        for rank, loc in enumerate(top_loc, 1):
            sent_idx = idxs[loc]
            rep_rows.append({
                "cluster": cid,
                "rank": rank,
                "score_tfidf": float(scores[loc]),
                "sentence": df.iloc[sent_idx][TEXT_COL]
            })

rep_df = pd.DataFrame(rep_rows)
rep_df.to_csv(OUTDIR/"cluster_representative_sentences.csv", index=False)
print("[INFO] 已导出：cluster_representative_sentences.csv")

# 6) 命名速查表（每簇一行：簇ID、规模、Top关键词、Top1代表句）
#    如果某簇代表句不足，留空
top1 = (rep_df.sort_values(["cluster","rank"])
              .groupby("cluster", as_index=False).first())

naming = (kw_df.merge(top1[["cluster","sentence"]], on="cluster", how="left")
               .sort_values("n_sentences", ascending=False))
naming.rename(columns={"sentence":"rep_sentence_top1"}, inplace=True)
naming.to_csv(OUTDIR/"cluster_naming_sheet.csv", index=False)

print("[INFO] 已导出：cluster_naming_sheet.csv")
print("\n完成。请使用 cluster_naming_sheet.csv 进行人工命名：")
print(naming.head(10).to_string(index=False))