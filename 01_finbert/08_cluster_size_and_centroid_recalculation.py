import pandas as pd
import numpy as np
from pathlib import Path

OUTDIR = Path("cluster_merge_outputs"); OUTDIR.mkdir(exist_ok=True, parents=True)
df2 = pd.read_parquet("sentences_with_cluster_p3_merged.parquet", engine="pyarrow")

# 新的簇大小统计
size_tbl2 = (
    df2.groupby("cluster_p3_merged")
       .size()
       .reset_index(name="n_sentences")
       .sort_values("n_sentences", ascending=False)
       .reset_index(drop=True)
)
size_tbl2["size_flag"] = np.where(
    size_tbl2["n_sentences"] < 150, "small",
    np.where(size_tbl2["n_sentences"] > 800, "large", "medium")
)
size_tbl2.to_csv(OUTDIR/"cluster_sizes_after_merge_final.csv", index=False)
print(size_tbl2.head(15).to_string(index=False))

# （可选）基于之前的 UMAP 簇心坐标做合并后的簇心（若你有 centers CSV）
centers = pd.read_csv(OUTDIR/"umap_cluster_centroids.csv")
centers["cluster"] = centers["cluster"].astype(str)

# 把中心点按映射聚合（均值）
map_df = pd.read_csv(OUTDIR/"cluster_merge_mapping.csv")
map_df["old_cluster"] = map_df["old_cluster"].astype(str)
map_df["new_cluster"] = map_df["new_cluster"].astype(str)
centers_m = centers.merge(map_df, left_on="cluster", right_on="old_cluster", how="left")
centers_m["new_cluster"] = centers_m["new_cluster"].fillna(centers_m["cluster"])

centers_new = (centers_m.groupby("new_cluster")[["x","y"]]
                        .mean()
                        .reset_index()
                        .rename(columns={"new_cluster":"cluster"}))
centers_new.to_csv(OUTDIR/"umap_cluster_centroids_merged.csv", index=False)
print("合并后簇心保存：", (OUTDIR/"umap_cluster_centroids_merged.csv").resolve())