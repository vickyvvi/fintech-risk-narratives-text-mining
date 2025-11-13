# ============================
# 0) 安装依赖（Colab）
# ============================
!pip -q install umap-learn hdbscan scikit-learn pandas pyarrow matplotlib

# ============================
# 1) 导入 & 配置
# ============================
import os
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

# 输入：含 'embedding' 列（list[float]）
INPUT_PATH = "/content/sentences_dedup_loose_keep_risky_with_sentiment.parquet"
OUTDIR = Path("/content/umap_runs")
OUTDIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "sentence_raw"
EMB_COL  = "embedding"           # 如果你是 emb_0...emb_n 展平的宽表，改用从列名收集
COLOR_BY = "finbert_label"       # 可换成 "Year" / "Company" 等

# ============================
# 2) 读数据 & 取出向量
# ============================
df = pd.read_parquet(INPUT_PATH)
print("读入规模:", df.shape)
if EMB_COL not in df.columns:
    # 兼容 emb_0...emb_n 宽表
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("找不到 embedding 列，也没有 emb_0... 列。请检查输入文件。")
    X = df[emb_cols].to_numpy(dtype=np.float32)
else:
    # 'embedding' 是 list；如果是字符串，需要转回 list
    if isinstance(df[EMB_COL].iloc[0], str):
        vecs = df[EMB_COL].apply(lambda s: ast.literal_eval(s))
    else:
        vecs = df[EMB_COL]
    X = np.vstack(vecs.values).astype(np.float32)

print("原向量维度:", X.shape)

# 可选：标准化（UMAP 对尺度不敏感，但对距离度量敏感。余弦时一般不需要标准化；欧式时可以试试）
# 这里我们不做标准化；如果你要做欧式度量，可取消注释：
# X = StandardScaler().fit_transform(X)

# ============================
# 3) UMAP 降维：32 / 64 / 128 维
# ============================
def run_umap(X, n_components=64, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42):
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )
    Z = reducer.fit_transform(X)
    return Z, reducer

umap_settings = [
    {"n_components": 32,  "metric": "cosine", "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 64,  "metric": "cosine", "n_neighbors": 15, "min_dist": 0.1},
    {"n_components": 128, "metric": "cosine", "n_neighbors": 15, "min_dist": 0.1},
]

all_outputs = {}
for cfg in umap_settings:
    Z, reducer = run_umap(X, **cfg)
    key = f"umap_{cfg['n_components']}d"
    all_outputs[key] = Z

    # 保存降维矩阵（.npy）与带列的数据表（parquet）
    np.save(OUTDIR / f"{key}.npy", Z)
    # 也可保存为列：z0..z{d-1}
    cols = {f"z{i}": Z[:, i] for i in range(cfg["n_components"])}
    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    out_df.to_parquet(OUTDIR / f"{key}.parquet", index=False)
    print(f"已保存：{OUTDIR / f'{key}.npy'}  和  {OUTDIR / f'{key}.parquet'}")

# ============================
# 4) 额外做一个 2D 可视化（便于 eyeballing）
# ============================
Z2, reducer2 = run_umap(X, n_components=2, metric="cosine", n_neighbors=15, min_dist=0.05)
vis_df = pd.DataFrame({"x": Z2[:,0], "y": Z2[:,1]})
if COLOR_BY in df.columns:
    vis_df[COLOR_BY] = df[COLOR_BY].astype(str).values
else:
    vis_df[COLOR_BY] = "NA"

plt.figure(figsize=(7,6), dpi=140)
for label, g in vis_df.groupby(COLOR_BY):
    plt.scatter(g["x"], g["y"], s=3, alpha=0.6, label=label)
plt.title(f"UMAP 2D (color by {COLOR_BY})")
plt.legend(markerscale=4, fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(OUTDIR / "umap_2d_scatter.png", dpi=160)
plt.show()

print("\n输出目录：", OUTDIR)