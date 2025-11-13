#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dedup_and_map.py  — 去重 + 映射 + 审查样本导出（修正版）
固定：sim_threshold=0.97；每簇抽取最不像的前10条（samples_per_cluster=10）。
输出：
  1) *_mapping.parquet      原句 -> 代表句映射（含 sim_to_rep）
  2) *_dedup.parquet        仅代表句（建模用）
  3) *_dedup_review.csv     代表句 + 簇内相似度统计 + 最不像成员
  4) *_cluster_samples.csv  每簇最不像的前10条成员样本
"""

import os, re, argparse, numpy as np, pandas as pd
from typing import List, Tuple, Optional

# ---------- IO ----------
def read_parquet_auto(path: str) -> pd.DataFrame:
    try: return pd.read_parquet(path, engine="pyarrow")
    except Exception: return pd.read_parquet(path, engine="fastparquet")

def write_parquet_auto(df: pd.DataFrame, path: str) -> None:
    for eng in ("pyarrow","fastparquet"):
        try: df.to_parquet(path, engine=eng, index=False); return
        except Exception: pass
    df.to_csv(path.rsplit(".",1)[0]+".csv", index=False, encoding="utf-8")
    print(f"[warn] parquet engines unavailable, wrote CSV fallback.")

# ---------- embedding columns (FIXED) ----------
def find_embedding_cols(df: pd.DataFrame) -> List[str]:
    # 1) 正常匹配 emb_0, emb_1, ...
    cols = [c for c in df.columns if re.fullmatch(r"emb_\d+", str(c).strip())]
    # 2) 兜底：容忍异常空格/类型
    if not cols:
        for c in df.columns:
            s = str(c).strip()
            if s.startswith("emb_") and s[4:].isdigit():
                cols.append(c)
    if not cols:
        first_cols = list(map(str, df.columns[:50]))
        raise ValueError(
            "No embedding columns like emb_0, emb_1 ... "
            f"(first 50 cols: {first_cols})"
        )
    # 排序
    def emb_idx(name: str) -> int:
        s = str(name).strip()
        return int(s.split("_", 1)[1])
    return sorted(cols, key=emb_idx)

# ---------- text & math ----------
def normalize_text_basic(s: str) -> str:
    if not isinstance(s,str): s = "" if s is None else str(s)
    s = re.sub(r"<[^>]+>", " ", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"[;,.!?]+$", "", s).strip()

def l2_normalize(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True); n = np.maximum(n, eps)
    return x / n

# ---------- exact dedup ----------
def exact_dedup(df: pd.DataFrame, text_col: str) -> Tuple[pd.Series, pd.Series]:
    norm = df[text_col].fillna("").astype(str).map(normalize_text_basic)
    first, rep = {}, []
    for i, s in zip(df.index, norm):
        if s not in first: first[s] = i
        rep.append(first[s])
    rep = pd.Series(rep, index=df.index)
    return rep, rep.eq(df.index)

# ---------- near-dup ----------
def near_dedup_via_faiss(emb: np.ndarray, sim_thr: float):
    try: import faiss  # type: ignore
    except Exception: return None, [], None
    X = l2_normalize(emb.astype("float32")); d = X.shape[1]
    try:
        res = faiss.StandardGpuResources(); index = faiss.GpuIndexFlatIP(res, d)
    except Exception:
        index = faiss.IndexFlatIP(d)
    index.add(X)
    n, B, k = X.shape[0], 4096, 64
    rep_of = -np.ones(n, dtype=np.int64)
    sim_to_rep = np.zeros(n, dtype=np.float32)
    assigned = np.zeros(n, dtype=bool)
    clusters = []
    for st in range(0,n,B):
        ed = min(n, st+B); sims, ids = index.search(X[st:ed], k)
        for li in range(ed-st):
            i = st+li
            if assigned[i]: continue
            rep = i; members=[]
            for sim,j in zip(sims[li], ids[li]):
                if j<0: continue
                if sim>=sim_thr and not assigned[j]:
                    assigned[j]=True; rep_of[j]=rep; sim_to_rep[j]=sim; members.append(j)
            clusters.append(members)
    for i in range(n):
        if rep_of[i]<0: rep_of[i]=i; sim_to_rep[i]=1.0; clusters.append([i])
    return rep_of, clusters, sim_to_rep

def near_dedup_via_sklearn(emb: np.ndarray, sim_thr: float):
    from sklearn.neighbors import NearestNeighbors
    X = l2_normalize(emb.astype("float32"))
    nbrs = NearestNeighbors(metric="cosine", algorithm="brute", radius=1.0-sim_thr, n_jobs=-1)
    nbrs.fit(X); dists_list, inds_list = nbrs.radius_neighbors(X, return_distance=True)
    n = X.shape[0]; assigned = np.zeros(n, dtype=bool)
    rep_of = -np.ones(n, dtype=np.int64); sim_to_rep = np.zeros(n, dtype=np.float32); clusters=[]
    for i in range(n):
        if assigned[i]: continue
        rep=i; members=[]
        for dist,j in sorted(zip(dists_list[i], inds_list[i]), key=lambda x:x[0]):
            sim=1.0-float(dist)
            if sim>=sim_thr and not assigned[j]:
                assigned[j]=True; rep_of[j]=rep; sim_to_rep[j]=sim; members.append(j)
        clusters.append(members)
    for i in range(n):
        if rep_of[i]<0: rep_of[i]=i; sim_to_rep[i]=1.0; clusters.append([i])
    return rep_of, clusters, sim_to_rep

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Deduplicate enriched sentence table and export mapping/review/samples.")
    ap.add_argument("--input", default="/content/risk_sentence_enriched.parquet")
    ap.add_argument("--output-prefix", default="/content/risk_sentence_enriched")
    ap.add_argument("--text-col", default="sentence_sub")
    ap.add_argument("--id-cols", default="sub_sent_id,sent_id,doc_id,Company,Year,FilingType,Sub-Sector")
    ap.add_argument("--sim-threshold", type=float, default=0.95)           # 将被覆盖
    ap.add_argument("--samples-per-cluster", type=int, default=0)          # 将被覆盖
    ap.add_argument("--sample-cap", type=int, default=0)
    args, _ = ap.parse_known_args()   # 兼容 notebook 注入的 -f 参数

    # 固定参数（方案B）
    args.sim_threshold = 0.97
    args.samples_per_cluster = 10
    print(f"[config] sim_threshold={args.sim_threshold}, samples_per_cluster={args.samples_per_cluster}")

    path = args.input; assert os.path.exists(path), f"Input not found: {path}"
    if path.endswith(".parquet"): df = read_parquet_auto(path)
    else:
        sep = "\t" if path.endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)
    if args.sample_cap and len(df)>args.sample_cap:
        df = df.iloc[:args.sample_cap].copy(); print(f"[info] sample_cap: {df.shape}")

    emb_cols = find_embedding_cols(df)
    print(f"[info] rows={len(df)}, emb_dim={len(emb_cols)}")

    # 1) exact 去重
    rep_text_idx, is_rep_text = exact_dedup(df, args.text_col)
    reps_text_idx = df.index[is_rep_text].tolist()
    print(f"[stat] exact unique = {len(reps_text_idx)} / {len(df)}")

    # 2) 近重复去重（在 exact unique 上）
    df_reps = df.loc[reps_text_idx].copy()
    reps_real_index = df_reps.index.to_numpy()
    E = df_reps[emb_cols].to_numpy(np.float32)
    rep_local, clusters_local, sims_local = near_dedup_via_faiss(E, args.sim_threshold)
    if rep_local is None:
        print("[info] FAISS unavailable -> sklearn.")
        rep_local, clusters_local, sims_local = near_dedup_via_sklearn(E, args.sim_threshold)

    near_rep_in_df = reps_real_index[rep_local]
    sims_to_rep_in_df = sims_local
    exact_rep_to_near_rep = pd.Series(near_rep_in_df, index=reps_real_index)

    # 3) 全量映射
    to_exact_rep = rep_text_idx
    final_rep_index = pd.Series(exact_rep_to_near_rep.reindex(to_exact_rep.values).to_numpy(), index=df.index)
    is_representative = final_rep_index.eq(df.index)

    sim_series_exact = pd.Series(sims_to_rep_in_df, index=reps_real_index)
    sim_to_rep = sim_series_exact.reindex(to_exact_rep.values).to_numpy()
    sim_to_rep = np.where(np.isnan(sim_to_rep), 1.0, sim_to_rep)

    rep_sizes = final_rep_index.value_counts().rename("cluster_size")
    cluster_size_series = final_rep_index.map(rep_sizes)

    # 4) 映射表
    keep_cols = []
    for c in [*filter(None, [x.strip() for x in args.id_cols.split(",")]),
              args.text_col, "sentence_raw", "sent_label", "hedge_ratio", "hedge_count",
              "prob_neg","prob_neu","prob_pos","token_len","char_len","number_count","has_year"]:
        if c in df.columns and c not in keep_cols: keep_cols.append(c)

    mapping = pd.DataFrame({
        "row_index": df.index,
        "rep_index": final_rep_index.values,
        "is_representative": is_representative.values,
        "sim_to_rep": sim_to_rep,
        "cluster_size": cluster_size_series.values,
    })
    mapping = pd.concat([mapping, df[keep_cols].reset_index(drop=True)], axis=1)

    # 5) 代表句
    reps_df = df.loc[is_representative].copy()
    reps_df["cluster_size"] = rep_sizes.reindex(reps_df.index).fillna(1).astype(int)

    # 6) 代表句审查表（簇内相似度统计 + 最不像成员）
    map_nonrep = mapping[mapping["row_index"] != mapping["rep_index"]]
    agg = map_nonrep.groupby("rep_index")["sim_to_rep"].agg(
        min_sim="min", max_sim="max", mean_sim="mean",
        p05_sim=lambda x: x.quantile(0.05), p25_sim=lambda x: x.quantile(0.25), p50_sim="median"
    )
    worst_member = map_nonrep.sort_values(["rep_index","sim_to_rep"]).groupby("rep_index").head(1).set_index("rep_index")["row_index"]

    reps_stats = reps_df.merge(agg, how="left", left_index=True, right_index=True)
    reps_stats[["min_sim","mean_sim"]] = reps_stats[["min_sim","mean_sim"]].fillna(1.0)
    reps_stats["worst_member_row_index"] = reps_stats.index.map(worst_member).astype("Int64")

    txt_col = "sentence_sub" if "sentence_sub" in df.columns else "sentence_raw"
    def pull(col): return reps_stats["worst_member_row_index"].apply(
        lambda i: df.loc[int(i), col] if pd.notna(i) and col in df.columns else pd.NA
    )
    for col in ["sub_sent_id","sent_id","Company","Year","sent_label","hedge_ratio","prob_neg","prob_neu","prob_pos"]:
        reps_stats[f"worst_member_{col}"] = pull(col)
    reps_stats[f"worst_member_{txt_col}"] = pull(txt_col)

    review_cols_pref = [
        "sub_sent_id","sent_id","doc_id","Company","Year","FilingType","Sub-Sector",
        txt_col,"sentence_raw",
        "sent_label","prob_neg","prob_neu","prob_pos",
        "hedge_count","hedge_ratio","token_len","char_len","number_count","has_year",
        "cluster_size","min_sim","mean_sim","p05_sim","p25_sim","p50_sim",
        "worst_member_row_index", f"worst_member_{txt_col}",
        "worst_member_sent_label","worst_member_prob_neg","worst_member_prob_neu","worst_member_prob_pos",
        "worst_member_hedge_ratio","worst_member_sub_sent_id","worst_member_sent_id",
        "worst_member_Company","worst_member_Year",
    ]
    review_cols = [c for c in review_cols_pref if c in reps_stats.columns]
    reps_review = reps_stats[review_cols].sort_values(["cluster_size","min_sim"], ascending=[False, True])

    # 7) 每簇最不像的前10条成员样本
    samples_df = None
    if args.samples_per_cluster > 0:
        tmp = map_nonrep.sort_values(["rep_index","sim_to_rep"])
        tmp["rank_in_cluster"] = tmp.groupby("rep_index")["sim_to_rep"].rank(method="first", ascending=True)
        sample_rows = tmp[tmp["rank_in_cluster"] <= args.samples_per_cluster].copy()
        cols_basic = ["sub_sent_id","sent_id","Company","Year","sent_label","hedge_ratio","prob_neg","prob_neu","prob_pos", txt_col]
        rep_side = df[cols_basic].add_prefix("rep_"); mem_side = df[cols_basic].add_prefix("mem_")
        sample_rows = sample_rows.join(rep_side, on="rep_index").join(mem_side, on="row_index")
        sample_cols = ["rep_index","row_index","cluster_size","sim_to_rep","rank_in_cluster"] + \
                      [c for c in sample_rows.columns if c.startswith("rep_")] + \
                      [c for c in sample_rows.columns if c.startswith("mem_")]
        samples_df = sample_rows[sample_cols].sort_values(["cluster_size","sim_to_rep"], ascending=[False, True])

    # 8) 写出
    base = args.output_prefix.rsplit(".",1)[0]
    write_parquet_auto(mapping, base + "_mapping.parquet")
    write_parquet_auto(reps_df, base + "_dedup.parquet")
    reps_review.to_csv(base + "_dedup_review.csv", index=False, encoding="utf-8")
    if samples_df is not None:
        samples_df.to_csv(base + "_cluster_samples.csv", index=False, encoding="utf-8")

    print(f"[ok] mapping: {base}_mapping.parquet (rows={len(mapping)})")
    print(f"[ok] dedup:   {base}_dedup.parquet   (rows={len(reps_df)})")
    print(f"[ok] review:  {base}_dedup_review.csv (rows={len(reps_review)})")
    if samples_df is not None:
        print(f"[ok] samples:{base}_cluster_samples.csv (rows={len(samples_df)})")

if __name__ == "__main__":
    main()