# === Merge assistant: Top-K → UMAP filter → final mapping ===
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- 参数 ----------
BASE = Path("cluster_merge_outputs")   # 你的结果目录
SIZE_CSV = BASE / "cluster_sizes.csv"
SIM_CSV  = BASE / "similarity_tfidf.csv"          # 如用嵌入相似度，改成 similarity_embeddings.csv
CENT_CSV = BASE / "umap_cluster_centroids.csv"    # UMAP 簇心坐标
TOP_K = 3
SIM_THRESHOLD = 0.60     # 语义相似度阈值（更保守可设 0.65/0.70）
UMAP_Q = 0.40            # UMAP 距离分位阈值（越小越严格，0.30~0.50 常用）

# ---------- 读取 & 规范化 ----------
size_tbl = pd.read_csv(SIZE_CSV)
sim_df   = pd.read_csv(SIM_CSV, index_col=0)
centers  = pd.read_csv(CENT_CSV)

# 找到簇列名，并统一为字符串
cid_col = [c for c in size_tbl.columns if "cluster" in c.lower()][0]
size_tbl[cid_col] = size_tbl[cid_col].astype(str)
sim_df.index = sim_df.index.astype(str); sim_df.columns = sim_df.columns.astype(str)
centers["cluster"] = centers["cluster"].astype(str)

# 安全去噪声（即使不存在 -1 也不报错）
sim_df = sim_df.drop(index="-1", columns="-1", errors="ignore")
size_tbl = size_tbl[size_tbl[cid_col] != "-1"].copy()
centers = centers[centers["cluster"] != "-1"].copy()

# 对齐共同簇集合（防止不一致）
common = sorted(set(sim_df.index) & set(sim_df.columns) & set(size_tbl[cid_col]) & set(centers["cluster"]))
sim_df = sim_df.loc[common, common]
size_tbl = size_tbl[size_tbl[cid_col].isin(common)].copy()
centers  = centers[centers["cluster"].isin(common)].copy()

# ---------- A. small → Top-K 候选（≥相似度阈值） ----------
small_ids  = size_tbl.query("size_flag=='small'")[cid_col].tolist()
target_ids = size_tbl.query("size_flag!='small'")[cid_col].tolist()

rows = []
for src in small_ids:
    if src not in sim_df.index: 
        continue
    sims = sim_df.loc[src, target_ids].drop(labels=[src], errors="ignore").sort_values(ascending=False)
    for tgt, val in sims.head(TOP_K).items():
        if pd.isna(val): 
            continue
        if float(val) >= SIM_THRESHOLD:
            trow = size_tbl.loc[size_tbl[cid_col]==tgt].iloc[0]
            rows.append({
                "source": src,
                "target": tgt,
                "similarity": round(float(val), 4),
                "target_size_flag": trow["size_flag"],
                "target_n_sentences": int(trow["n_sentences"])
            })

topk_df = pd.DataFrame(rows).sort_values(["source","similarity"], ascending=[True,False])
out_topk = BASE / "merge_candidates_topk.csv"
topk_df.to_csv(out_topk, index=False)

print(f"[A] small={len(small_ids)}，Top-{TOP_K} 候选（≥{SIM_THRESHOLD}）数量：{topk_df.shape[0]}")
print("示例：")
print(topk_df.head(10).to_string(index=False))
print("已保存：", out_topk.resolve())

# ---------- B. UMAP 距离过滤（语义+空间都近） ----------
# 建立坐标字典
pos = centers.set_index("cluster")[["x","y"]].to_dict(orient="index")

def umap_dist(a, b):
    xa, ya = pos[a]["x"], pos[a]["y"]
    xb, yb = pos[b]["x"], pos[b]["y"]
    return float(((xa - xb)**2 + (ya - yb)**2) ** 0.5)

if not topk_df.empty:
    topk_df["umap_dist"] = topk_df.apply(lambda r: umap_dist(str(r["source"]), str(r["target"])), axis=1)
    thr = topk_df["umap_dist"].quantile(UMAP_Q)  # 分位数阈值
    filtered = (topk_df[topk_df["umap_dist"] <= thr]
                .sort_values(["source","similarity"], ascending=[True,False])
                .reset_index(drop=True))
else:
    thr = np.nan
    filtered = topk_df.copy()

out_filtered = BASE / "merge_candidates_topk_umapfiltered.csv"
filtered.to_csv(out_filtered, index=False)

print(f"\n[B] UMAP 距离过滤阈值（分位）: {UMAP_Q:.2f} → 数值阈值 {thr:.3f}（越小越近）")
print(f"通过过滤的候选数量：{filtered.shape[0]}")
print("示例：")
print(filtered.head(10).to_string(index=False))
print("已保存：", out_filtered.resolve())

# ---------- C. 生成最终合并映射（每个 small 仅保留一个 target） ----------
# 若 B 结果为空，则退回 A 结果做映射
base_df = filtered if not filtered.empty else topk_df

# 每个 small 仅取相似度最高的 target
if not base_df.empty:
    best = (base_df.sort_values(["source","similarity"], ascending=[True,False])
                  .groupby("source", as_index=False).first())
else:
    best = pd.DataFrame(columns=["source","target"])

# 构建映射：small -> target；其他保持不变
all_ids = size_tbl[cid_col].astype(str).unique().tolist()
mapping = {c:c for c in all_ids}
for _, r in best.iterrows():
    mapping[str(r["source"])] = str(r["target"])

map_df = pd.DataFrame({"old_cluster": list(mapping.keys()),
                       "new_cluster": list(mapping.values())}).sort_values("old_cluster")

out_map = BASE / "cluster_merge_mapping.csv"
map_df.to_csv(out_map, index=False)
print("\n[C] 合并映射已保存：", out_map.resolve())
print("映射示例：")
print(map_df.head(12).to_string(index=False))

# 合并后簇大小预览
merged_size = (size_tbl.assign(new_cluster=size_tbl[cid_col].map(mapping))
                        .groupby("new_cluster", as_index=False)["n_sentences"].sum()
                        .sort_values("n_sentences", ascending=False))
out_size = BASE / "cluster_sizes_after_merge_preview.csv"
merged_size.to_csv(out_size, index=False)
print("\n合并后簇大小预览（Top 15）：")
print(merged_size.head(15).to_string(index=False))
print("已保存：", out_size.resolve())