# ============================
# 0) 依赖（上面已装过 hdbscan / sklearn）
# ============================
import numpy as np
import pandas as pd
from pathlib import Path
import hdbscan

IN_DIR = Path("/content/umap_runs")
REDUCED_FILE = IN_DIR / "umap_64d.parquet"   # ← 改成 32/64/128 任一个
OUT_DIR = Path("/content/hdbscan_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 读入包含 z0..zd-1 的 parquet
df_red = pd.read_parquet(REDUCED_FILE)
z_cols = [c for c in df_red.columns if c.startswith("z")]
Z = df_red[z_cols].to_numpy(dtype=np.float32)
print("用于聚类的降维矩阵:", Z.shape)

# 4 组参数（可自行调整）
param_grid = [
    {"min_cluster_size":  50, "min_samples": None, "cluster_selection_epsilon": 0.00, "metric": "euclidean"},
    {"min_cluster_size": 100, "min_samples": None, "cluster_selection_epsilon": 0.00, "metric": "euclidean"},
    {"min_cluster_size": 100, "min_samples":  50, "cluster_selection_epsilon": 0.00, "metric": "euclidean"},
    {"min_cluster_size": 200, "min_samples": 100, "cluster_selection_epsilon": 0.05, "metric": "euclidean"},
]

# 逐组跑 HDBSCAN
summaries = []
labels_cols = []

for i, p in enumerate(param_grid, 1):
    print(f"\n=== HDBSCAN 方案{i}: {p} ===")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=p["min_cluster_size"],
        min_samples=p["min_samples"],
        cluster_selection_epsilon=p["cluster_selection_epsilon"],
        metric=p["metric"],
        core_dist_n_jobs= -1,               # 多线程
        prediction_data=True
    )
    labels = clusterer.fit_predict(Z)      # -1 为噪声
    col = f"hdbscan_{i}_label"
    df_red[col] = labels
    labels_cols.append(col)

    # 摘要
    n_noise = int((labels == -1).sum())
    n_assigned = int((labels != -1).sum())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_ratio = round(n_noise / len(labels), 4)

    # 软成员概率（0-1），可用于后续过滤低置信度样本
    if hasattr(clusterer, "probabilities_"):
        df_red[f"hdbscan_{i}_prob"] = clusterer.probabilities_
    else:
        df_red[f"hdbscan_{i}_prob"] = np.nan

    # 保存该方案的完整结果
    out_path = OUT_DIR / f"hdbscan_solution_{i}.parquet"
    df_red[[*z_cols, col, f"hdbscan_{i}_prob"]].to_parquet(out_path, index=False)

    summaries.append({
        "solution": i,
        **p,
        "n_clusters": n_clusters,
        "n_assigned": n_assigned,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio
    })
    print(f"  -> clusters={n_clusters}, assigned={n_assigned}, noise={n_noise} ({noise_ratio*100:.1f}%)  已保存: {out_path}")

# 汇总表
summary_df = pd.DataFrame(summaries)
summary_df.to_csv(OUT_DIR / "hdbscan_summaries.csv", index=False)

# 把 4 个方案的 label 也拼到一份总表（带原元数据/文本，便于后续分析）
keep_cols = [c for c in df_red.columns if not c.startswith("z")]  # 保留非z列（原元数据、文本）
final_df = df_red[keep_cols + labels_cols].copy()
final_df.to_parquet(OUT_DIR / "hdbscan_all_solutions_with_meta.parquet", index=False)

print("\n完成 ✅")
print("摘要：", OUT_DIR / "hdbscan_summaries.csv")
print("总表：", OUT_DIR / "hdbscan_all_solutions_with_meta.parquet")