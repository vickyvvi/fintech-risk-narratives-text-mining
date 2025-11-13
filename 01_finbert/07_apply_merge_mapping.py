import pandas as pd
from pathlib import Path

# 路径配置
INPUT_PATH = "/content/sentences_with_sentiment_plus_mpnet_no384_umap128_gpu_hdbscan_p3.parquet"  # 改成你的
BASE = Path("cluster_merge_outputs")
MAP_CSV = BASE / "cluster_merge_mapping.csv"          # 上一步生成的映射
OUT_PARQUET = "sentences_with_cluster_p3_merged.parquet"  # 输出文件

# 读取
df = pd.read_parquet(INPUT_PATH, engine="pyarrow")
mapping = pd.read_csv(MAP_CSV)

# 规范化 id 为字符串，构造字典
mapping["old_cluster"] = mapping["old_cluster"].astype(str)
mapping["new_cluster"] = mapping["new_cluster"].astype(str)
map_dict = dict(zip(mapping["old_cluster"], mapping["new_cluster"]))

# 应用映射（保留 -1：若原数据有 -1，映射表里没有就保持 -1 不变）
df["cluster_p3_str"] = df["cluster_p3"].astype(str)
df["cluster_p3_merged"] = df["cluster_p3_str"].map(map_dict).fillna(df["cluster_p3_str"])

# 保存
df.to_parquet(OUT_PARQUET, engine="pyarrow", index=False)
print("已保存：", OUT_PARQUET)