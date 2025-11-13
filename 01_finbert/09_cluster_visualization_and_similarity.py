# A) UMAP簇心分布（GPU优先）
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "/content/sentences_with_sentiment_plus_mpnet_no384_umap128_gpu_hdbscan_p3.parquet"
CLUSTER_COL = "cluster_p3"
TEXT_COL = "sentence_raw"  # 不直接用，但保留
EMB_PREFIX = "emb_"
OUTDIR = Path("cluster_merge_outputs"); OUTDIR.mkdir(exist_ok=True)

# 读取：优先 GPU（cudf），失败用 pandas
try:
    import cudf
    gdf = cudf.read_parquet(INPUT_PATH)
    df = gdf.to_pandas()
    USE_GPU = True
    print("[INFO] loaded via cudf (GPU).")
except Exception:
    import pandas as pd
    df = pd.read_parquet(INPUT_PATH, engine="pyarrow")
    USE_GPU = False
    print("[INFO] loaded via pandas (CPU).")

assert CLUSTER_COL in df.columns
has_xy = ("umap2_x" in df.columns) and ("umap2_y" in df.columns)

# 簇心（如果已有umap2_x/umap2_y，直接均值）
if has_xy:
    centers = (
        df.groupby(CLUSTER_COL)[["umap2_x","umap2_y"]]
          .mean()
          .reset_index()
          .rename(columns={"umap2_x":"x","umap2_y":"y", CLUSTER_COL:"cluster"})
    )
else:
    # 没有umap坐标：若有嵌入则先算“簇心嵌入”，再做UMAP到2D
    emb_cols = [c for c in df.columns if c.startswith(EMB_PREFIX)]
    if not emb_cols:
        raise RuntimeError("未检测到 umap2_x/umap2_y，也没有嵌入列 emb_*，无法做UMAP簇心图。")
    # 计算簇心嵌入
    emb_mat = df[emb_cols].to_numpy(dtype=np.float32)
    clusters = sorted(df[CLUSTER_COL].unique().tolist())
    centroids = []
    for cid in clusters:
        idx = (df[CLUSTER_COL].values == cid)
        centroids.append(emb_mat[idx].mean(axis=0))
    centroids = np.vstack(centroids)

    # UMAP 降维（GPU→cuml；无则CPU→umap-learn）
    try:
        from cuml.manifold import UMAP as cuUMAP
        import cupy as cp
        reducer = cuUMAP(n_components=2, random_state=42)
        umap_2d = reducer.fit_transform(cp.asarray(centroids))
        xy = cp.asnumpy(umap_2d)
        print("[INFO] UMAP on GPU (cuml).")
    except Exception:
        import umap  # pip install umap-learn
        reducer = umap.UMAP(n_components=2, random_state=42)
        xy = reducer.fit_transform(centroids)
        print("[INFO] UMAP on CPU (umap-learn).")

    centers = pd.DataFrame({"cluster": clusters, "x": xy[:,0], "y": xy[:,1]})

# 叠加簇大小标记
size_tbl = df.groupby(CLUSTER_COL).size().rename("n_sentences").reset_index()
size_tbl["size_flag"] = np.where(
    size_tbl["n_sentences"] < 150, "small",
    np.where(size_tbl["n_sentences"] > 800, "large", "medium")
)
centers = centers.merge(size_tbl, left_on="cluster", right_on=CLUSTER_COL, how="left")

plt.figure(figsize=(8,6))
ax = sns.scatterplot(
    data=centers, x="x", y="y",
    hue="size_flag", size="n_sentences",
    sizes=(50, 350), alpha=0.9
)
for _, r in centers.iterrows():
    ax.text(r["x"], r["y"], str(r["cluster"]), fontsize=8, ha="center", va="center")
plt.title("Cluster Centroids in UMAP Space")
plt.tight_layout()
plt.savefig(OUTDIR / "umap_cluster_centroids.png", dpi=180)
plt.show()

centers.to_csv(OUTDIR / "umap_cluster_centroids.csv", index=False)
print("Saved:", (OUTDIR / "umap_cluster_centroids.png").resolve())

# B) 簇间相似度矩阵 + 层次聚类树状图（嵌入优先，无嵌入用TF-IDF）
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

INPUT_PATH = "/content/sentences_with_sentiment_plus_mpnet_no384_umap128_gpu_hdbscan_p3.parquet"
CLUSTER_COL = "cluster_p3"
TEXT_COL = "sentence_raw"
EMB_PREFIX = "emb_"
OUTDIR = Path("cluster_merge_outputs"); OUTDIR.mkdir(exist_ok=True)

# 读
try:
    import cudf
    df = cudf.read_parquet(INPUT_PATH).to_pandas()
except Exception:
    df = pd.read_parquet(INPUT_PATH, engine="pyarrow")

assert CLUSTER_COL in df.columns and TEXT_COL in df.columns
clusters = sorted(df[CLUSTER_COL].unique().tolist())
size_tbl = df.groupby(CLUSTER_COL).size().rename("n_sentences").reset_index()
size_tbl["size_flag"] = np.where(
    size_tbl["n_sentences"] < 150, "small",
    np.where(size_tbl["n_sentences"] > 800, "large", "medium")
)

# 优先：用嵌入簇心余弦相似度
emb_cols = [c for c in df.columns if c.startswith(EMB_PREFIX)]
sim_mat = None
sim_source = None

if emb_cols:
    emb = df[emb_cols].to_numpy(dtype=np.float32)
    centroids = []
    for cid in clusters:
        idx = (df[CLUSTER_COL].values == cid)
        centroids.append(emb[idx].mean(axis=0))
    centroids = np.vstack(centroids)
    # L2 归一化 + 余弦
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sim_mat = centroids @ centroids.T
    np.fill_diagonal(sim_mat, 1.0)
    sim_source = "embeddings"
else:
    # 退而求其次：TF-IDF 簇文档质心余弦
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    cluster_docs = (
        df.groupby(CLUSTER_COL)[TEXT_COL]
          .apply(lambda s: "\n".join(map(str, s)))
          .reindex(clusters)
    )
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(cluster_docs.values)        # (n_clusters, vocab) 稀疏
    Xn = normalize(X, norm="l2", axis=1)
    sim_mat = (Xn @ Xn.T).A                           # 稀疏→稠密
    np.fill_diagonal(sim_mat, 1.0)
    sim_source = "tfidf"

# 保存相似度矩阵
sim_df = pd.DataFrame(sim_mat, index=clusters, columns=clusters)
sim_csv = OUTDIR / f"similarity_{sim_source}.csv"
sim_df.to_csv(sim_csv)
print("Saved:", sim_csv.resolve())

# 构造层次聚类树状图
def sim_to_condensed_distance(S):
    D = 1.0 - S
    D = np.clip(D, 0.0, 1.0)
    return squareform(D, checks=False)

Z = linkage(sim_to_condensed_distance(sim_mat), method="average")

labels_map = {
    cid: f"{cid} (n={int(size_tbl.loc[size_tbl[CLUSTER_COL]==cid,'n_sentences'].values[0])},"
         f"{size_tbl.loc[size_tbl[CLUSTER_COL]==cid,'size_flag'].values[0]})"
    for cid in clusters
}

plt.figure(figsize=(12,6))
dendrogram(Z, labels=[labels_map[c] for c in clusters], leaf_rotation=90, leaf_font_size=8)
plt.title(f"Cluster Merge Dendrogram (source: {sim_source})")
plt.tight_layout()
plt.savefig(OUTDIR / "dendrogram.png", dpi=180)
plt.show()
print("Saved:", (OUTDIR / "dendrogram.png").resolve())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 读簇心表
centers = pd.read_csv("cluster_merge_outputs/umap_cluster_centroids.csv")

# 过滤掉 cluster == -1
centers = centers[centers["cluster"] != -1]

plt.figure(figsize=(8,6))
ax = sns.scatterplot(
    data=centers,
    x="x", y="y",
    hue="size_flag",
    size="n_sentences",
    sizes=(50, 350),
    alpha=0.9
)

for _, r in centers.iterrows():
    ax.text(r["x"], r["y"], str(r["cluster"]), fontsize=8,
            ha="center", va="center")

plt.title("Cluster Centroids in UMAP Space (without noise)")
plt.tight_layout()

out_path = Path("cluster_merge_outputs/umap_cluster_centroids_no_noise.png")
plt.savefig(out_path, dpi=180)
plt.show()

print("已保存：", out_path.resolve())