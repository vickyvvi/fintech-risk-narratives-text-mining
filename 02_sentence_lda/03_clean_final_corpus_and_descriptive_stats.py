import os
import pandas as pd
from collections import Counter

# ========= 1) 写入额外停用词（unigram） =========
extra_stop_unigrams = [
    "include","result","subject","certain","addition",
    "future","new","provide","business","company"
]

extra_path = "/content/stopwords_extra_unigram.txt"
with open(extra_path, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(set(w.lower() for w in extra_stop_unigrams))))
print(f"[ok] extra stopwords saved -> {extra_path}")

# ========= 2) 读取当前 tokens 表 =========
in_path  = "/content/sentlda_stage/sentlda_phrases.parquet"
out_path = "/content/sentlda_stage/sentlda_final.parquet"
assert os.path.exists(in_path), f"not found: {in_path}"

df = pd.read_parquet(in_path)
assert "tokens_final" in df.columns, "tokens_final column not found."

# ========= 3) 清理：仅移除这些 unigram（不动短语）=========
stop_set = set(extra_stop_unigrams)

def drop_unigram_stops(tokens):
    out = []
    for t in tokens:
        # 只过滤完全匹配的 unigram；保留任何带下划线的 n-gram
        if "_" not in t and t.lower() in stop_set:
            continue
        out.append(t)
    return out

# 统计清理前
all_tokens_before = [t for ts in df["tokens_final"] for t in ts]
uni_before = sum(1 for t in all_tokens_before if "_" not in t)
ng_before  = sum(1 for t in all_tokens_before if "_" in t)

# 执行清理
df["tokens_final"] = df["tokens_final"].apply(drop_unigram_stops)
df["tokens_final_str"] = df["tokens_final"].apply(lambda ts: " ".join(map(str, ts)))

# 统计清理后
all_tokens_after = [t for ts in df["tokens_final"] for t in ts]
uni_after = sum(1 for t in all_tokens_after if "_" not in t)
ng_after  = sum(1 for t in all_tokens_after if "_" in t)

# ========= 4) 保存最新总表 =========
df.to_parquet(out_path, index=False)
print(f"[ok] latest table saved -> {out_path}")

# ========= 5) 打印对比与快速检查 =========
print("\n===== STATS (overall tokens) =====")
print(f"rows: {len(df)}")
print(f"unigram  before/after: {uni_before:,}  ->  {uni_after:,}  (Δ {uni_after-uni_before:+,})")
print(f"n-gram   before/after: {ng_before:,}   ->  {ng_after:,}   (Δ {ng_after-ng_before:+,})")
print(f"share n-gram after: {ng_after / max(1,(uni_after+ng_after)):.2%}")

cnt_before = Counter(all_tokens_before)
cnt_after  = Counter(all_tokens_after)

print("\nTop 20 tokens BEFORE:")
for w,c in cnt_before.most_common(20):
    print(f"{w:25s} {c}")

print("\nTop 20 tokens AFTER:")
for w,c in cnt_after.most_common(20):
    print(f"{w:25s} {c}")

# 额外：分别看前10个短语/单词（AFTER）
print("\nTop 10 phrases (AFTER):")
k = 0
for w,c in cnt_after.most_common():
    if "_" in w:
        print(f"{w:25s} {c}")
        k += 1
        if k >= 10: break

print("\nTop 10 unigrams (AFTER):")
k = 0
for w,c in cnt_after.most_common():
    if "_" not in w:
        print(f"{w:25s} {c}")
        k += 1
        if k >= 10: break

import pandas as pd
from collections import Counter

# ========= 配置 =========
in_path  = "/content/sentlda_stage/sentlda_final.parquet"
out_path = "/content/sentlda_stage/sentlda_final_v2.parquet"

extra_stop_unigrams_v2 = [
    "change","service","operation","increase","impact","time","example", "additional", "ability", "experience",
    "market","issue","affect","activity","experience","share","respect","operate","use","party","states","relate","continue","product","united","requirement","cause","significant",
    "exist","base","expect","event","far"
]

stop_set = set(extra_stop_unigrams_v2)

# ========= 读取 =========
df = pd.read_parquet(in_path)
assert "tokens_final" in df.columns, "tokens_final column not found."

# ========= 清理函数 =========
def drop_extra_unigrams(tokens):
    out = []
    for t in tokens:
        if "_" not in t and t.lower() in stop_set:
            continue
        out.append(t)
    return out

# ========= 清理前统计 =========
all_tokens_before = [t for ts in df["tokens_final"] for t in ts]
cnt_before = Counter(all_tokens_before)
print("Top 10 unigrams BEFORE:")
for w,c in [(w,c) for w,c in cnt_before.most_common() if "_" not in w][:10]:
    print(f"{w:20s} {c}")

# ========= 应用清理 =========
df["tokens_final"] = df["tokens_final"].apply(drop_extra_unigrams)
df["tokens_final_str"] = df["tokens_final"].apply(lambda ts: " ".join(ts))

# ========= 清理后统计 =========
all_tokens_after = [t for ts in df["tokens_final"] for t in ts]
cnt_after = Counter(all_tokens_after)

print("\nTop 30 unigrams AFTER:")
for w,c in [(w,c) for w,c in cnt_after.most_common() if "_" not in w][:30]:
    print(f"{w:20s} {c}")

print("\nTop 30 phrases AFTER:")
for w,c in [(w,c) for w,c in cnt_after.most_common() if "_" in w][:30]:
    print(f"{w:20s} {c}")

# ========= 保存 =========
df.to_parquet(out_path, index=False)
print(f"\n[ok] saved cleaned table -> {out_path}")



import pandas as pd

path = "/content/sentlda_stage/sentlda_final_v2.parquet"

# 只读取列名和前几行
df = pd.read_parquet(path)

print("===== 列名 =====")
print(df.columns.tolist())

print("\n===== 前几行 (只显示关键列) =====")
print(df[["sub_sent_id","sentence_sub","tokens_final","tokens_final_str"]].head(5))



import os, ast
import numpy as np
import pandas as pd
from collections import Counter

BASE = "/content/sentlda_stage"
INP  = f"{BASE}/sentlda_final_v2.parquet"
OUT_NOEMPTY = f"{BASE}/sentlda_final_v2_noempty.parquet"
OUT_MAP     = f"{BASE}/sentlda_final_v2_mapping.csv"

df = pd.read_parquet(INP)
print("[info] columns:", list(df.columns))

def is_nan_scalar(x) -> bool:
    """只对标量做 NaN 判断；数组/列表一律返回 False。"""
    # 标量 float('nan') 或 numpy.nan
    if isinstance(x, float):
        return np.isnan(x)
    # numpy scalar
    if isinstance(x, (np.floating, np.integer)):
        try:
            return bool(np.isnan(x))
        except Exception:
            return False
    return False

def to_list_tokens(x):
    """
    统一把 x 转为 List[str]:
      - list/tuple -> 逐个转 str，去空
      - numpy.ndarray -> 展平后逐个转 str
      - "['a','b']" -> ast.literal_eval
      - "a b c"     -> .split()
      - None/NaN    -> []
      - 其他对象     -> 尝试 list(...) 再转 str
    """
    # None 或 NaN 标量
    if x is None or is_nan_scalar(x):
        return []
    # 已经是 list/tuple
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x if str(t).strip()]
    # numpy 数组
    if isinstance(x, np.ndarray):
        return [str(t) for t in x.ravel().tolist() if str(t).strip()]
    # 字符串：尝试解析或按空格切
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # 形如 "['a','b']" / "('a','b')" 就解析
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple, np.ndarray)):
                    seq = val.tolist() if isinstance(val, np.ndarray) else val
                    return [str(t) for t in seq if str(t).strip()]
            except Exception:
                # 解析失败则按空格分
                pass
        # 默认空格分
        return [w for w in s.split() if w]
    # 其他可迭代
    try:
        return [str(t) for t in list(x) if str(t).strip()]
    except Exception:
        return []

# 如果没有 tokens_final，但有 tokens_final_str，则用后者修复
if "tokens_final" not in df.columns and "tokens_final_str" in df.columns:
    print("[warn] tokens_final 不存在，使用 tokens_final_str 修复。")
    df["tokens_final"] = df["tokens_final_str"].apply(to_list_tokens)
else:
    df["tokens_final"] = df["tokens_final"].apply(to_list_tokens)

# 同步 tokens_final_str（以免不一致）
df["tokens_final_str"] = df["tokens_final"].apply(lambda ts: " ".join(ts))

# 统计空文档
df = df.reset_index(drop=False).rename(columns={"index":"orig_row"})
df["token_len"] = df["tokens_final"].apply(len)

num_all = len(df)
num_empty = int((df["token_len"] == 0).sum())
print(f"[stat] total={num_all}, empty={num_empty}, nonempty={num_all - num_empty}")

print("\n[sample tokens_final（前3行）]:")
print(df["tokens_final"].head(3).tolist())

# 仅在建模视图中过滤空文档
df_model = df[df["token_len"] > 0].copy().reset_index(drop=True)
df_model["lda_doc_id"] = range(len(df_model))

# 保存建模视图与映射
os.makedirs(BASE, exist_ok=True)
df_model.to_parquet(OUT_NOEMPTY, index=False)
map_cols_hint = [c for c in ["lda_doc_id","orig_row","sub_sent_id","sent_id","sentence_sub"] if c in df_model.columns]
df_model[map_cols_hint].to_csv(OUT_MAP, index=False)

print(f"\n[ok] saved model view: {OUT_NOEMPTY} (rows={len(df_model)})")
print(f"[ok] saved mapping:    {OUT_MAP}")

# 词表规模（快速 sanity check）
all_tokens = [t for ts in df_model["tokens_final"] for t in ts]
print(f"[stat] vocab_size≈ {len(set(all_tokens))}, tokens_total= {len(all_tokens)}")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
describe_corpus_compare.py

同时生成：
- 去重前（原始分句表）的描述性统计
- 去重后（LDA输入表）的描述性统计
- 二者对比汇总 + 重复度（若提供mapping）

输入参数（按你的环境修改下面 DEFAULTS）：
  ORIGINAL_PATH : 去重前大表（如 risk_sentence_enriched.parquet 或 CSV/TSV）
  DEDUP_PATH    : 去重后、已准备好 tokens_final 的表（sentlda_final_v2_noempty.parquet）
  MAPPING_PATH  : （可选）你之前导出的 mapping（含 cluster_size），CSV/Parquet 都行

输出到 OUTDIR：
  - full_corpus_stats.csv
  - dedup_corpus_stats.csv
  - compare_full_vs_dedup.csv
  - dup_stats.csv（若提供mapping：重复簇统计）
  - 若 DEDUP 有 tokens_final：top_tokens / top_phrases、句长直方图
"""

from pathlib import Path
import os, re, ast
from typing import List
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 默认路径（按需修改）======
ORIGINAL_PATH = Path("/content/risk_sentence_enriched.parquet")         # 你的去重前大表
DEDUP_PATH    = Path("/content/sentlda_stage/sentlda_final_v2_noempty.parquet") # 你的去重后表
MAPPING_PATH  = Path("/content/risk_sentence_enriched_mapping.parquet") # 可留空/不存在也行
OUTDIR        = Path("/content/sentlda_stage/describe_compare"); OUTDIR.mkdir(parents=True, exist_ok=True)

# 列名
TEXT_COL_ORI  = "sentence_sub"     # 原始表里存句子的列
TEXT_COL_DED  = "tokens_final"     # 去重后表里存 LDA token 的列（list[str] 或字符串）
ID_COL        = "sub_sent_id"      # 若存在用于预览

# --------- 读表（支持 parquet/csv/tsv）---------
def read_any(path: Path) -> pd.DataFrame:
    assert path.exists(), f"Not found: {path}"
    if path.suffix.lower() == ".parquet":
        for eng in ("pyarrow","fastparquet"):
            try:
                return pd.read_parquet(path, engine=eng)
            except Exception:
                pass
        raise RuntimeError("Parquet engine missing")
    # csv/tsv
    sep = "\t" if path.suffix.lower()==".tsv" else ","
    return pd.read_csv(path, sep=sep)

# --------- tokens_final 统一为 list[str] ----------
def to_list_tokens(x) -> List[str]:
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
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple, np.ndarray)):
                    seq = val.tolist() if isinstance(val, np.ndarray) else val
                    return [str(t).strip() for t in seq if str(t).strip()]
            except Exception:
                pass
        # 退化为空格分词（若已是 "a b c"）
        return [w for w in s.split() if w.strip()]
    try:
        return [str(t).strip() for t in list(x) if str(t).strip()]
    except Exception:
        return []

# --------- 简单 tokenizer（用于原始表做句长/词频）---------
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|<NUM>")
def simple_tokenize_text(s: str) -> List[str]:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = re.sub(r"<[^>]+>", " ", s)  # 去HTML
    s = s.lower()
    return WORD_RE.findall(s)

# --------- 统计打印为 DataFrame ----------
def basic_stats_from_texts(texts: List[str]) -> pd.DataFrame:
    lens_char = [len(t) for t in texts]
    tokens = [simple_tokenize_text(t) for t in texts]
    lens_tok = [len(ts) for ts in tokens]
    # 词表
    vocab = set()
    for ts in tokens: vocab.update(ts)
    rows = [
        ("#Documents (sentences)", len(texts)),
        ("#Non-empty documents", int(np.sum([l>0 for l in lens_tok]))),
        ("#Empty documents",     int(np.sum([l==0 for l in lens_tok]))),
        ("Vocabulary size (rough)", len(vocab)),
        ("Avg tokens per sentence", round(float(np.mean(lens_tok)) if len(texts) else 0.0, 2)),
        ("Min / P50 / P90 / P99 / Max length",
         f"{int(np.min(lens_tok)) if lens_tok else 0} / "
         f"{float(np.percentile(lens_tok,50)) if lens_tok else 0:.0f} / "
         f"{float(np.percentile(lens_tok,90)) if lens_tok else 0:.0f} / "
         f"{float(np.percentile(lens_tok,99)) if lens_tok else 0:.0f} / "
         f"{int(np.max(lens_tok)) if lens_tok else 0}")
    ]
    return pd.DataFrame(rows, columns=["Statistic","Value"]), tokens

def stats_from_tokens_list(tokens_list: List[List[str]]) -> pd.DataFrame:
    lens_tok = [len(ts) for ts in tokens_list]
    vocab = set()
    for ts in tokens_list: vocab.update(ts)
    rows = [
        ("#Documents (sentences)", len(tokens_list)),
        ("#Non-empty documents", int(np.sum([l>0 for l in lens_tok]))),
        ("#Empty documents",     int(np.sum([l==0 for l in lens_tok]))),
        ("Vocabulary size (all tokens)", len(vocab)),
        ("Avg tokens per sentence", round(float(np.mean(lens_tok)) if len(tokens_list) else 0.0, 2)),
        ("Min / P50 / P90 / P99 / Max length",
         f"{int(np.min(lens_tok)) if lens_tok else 0} / "
         f"{float(np.percentile(lens_tok,50)) if lens_tok else 0:.0f} / "
         f"{float(np.percentile(lens_tok,90)) if lens_tok else 0:.0f} / "
         f"{float(np.percentile(lens_tok,99)) if lens_tok else 0:.0f} / "
         f"{int(np.max(lens_tok)) if lens_tok else 0}")
    ]
    return pd.DataFrame(rows, columns=["Statistic","Value"])

# ================== 主流程 ==================
# 1) 去重前（原始表）
df_full = read_any(ORIGINAL_PATH)
assert TEXT_COL_ORI in df_full.columns, f"Column '{TEXT_COL_ORI}' not in original table."
texts_full = df_full[TEXT_COL_ORI].fillna("").astype(str).tolist()
stats_full_df, tokens_full = basic_stats_from_texts(texts_full)

# 精确去重（大小写/空白规整后）
norm = [re.sub(r"\s+"," ", t.strip().lower()) for t in texts_full]
n_total = len(norm)
n_unique = len(pd.unique(pd.Series(norm)))
dup_rate = 1 - n_unique / n_total if n_total else 0.0

# 2) 去重后（LDA输入表）
df_dedup = read_any(DEDUP_PATH)
assert TEXT_COL_DED in df_dedup.columns, f"Column '{TEXT_COL_DED}' not in dedup table."
tokens_ded = [to_list_tokens(x) for x in df_dedup[TEXT_COL_DED].tolist()]
stats_ded_df = stats_from_tokens_list(tokens_ded)

# 3) 保存各自统计
stats_full_df.to_csv(OUTDIR / "full_corpus_stats.csv", index=False)
stats_ded_df.to_csv(OUTDIR / "dedup_corpus_stats.csv", index=False)

# 4) 对比表
compare_rows = [
    ("Total sentences (before)", n_total),
    ("Unique sentences (exact, lower/trim)", n_unique),
    ("Duplicate rate (exact)", round(dup_rate, 4)),
    ("Sentences after dedup (LDA input)", len(tokens_ded)),
]
compare_df = pd.DataFrame(compare_rows, columns=["Metric","Value"])
compare_df.to_csv(OUTDIR / "compare_full_vs_dedup.csv", index=False)

# 若有 mapping（带 cluster_size），输出重复簇分布
if MAPPING_PATH.exists():
    dm = read_any(MAPPING_PATH)
    if "cluster_size" in dm.columns:
        cs = dm["cluster_size"].astype(int).tolist()
        dup_stats = pd.DataFrame({
            "count_docs":[len(cs)],
            "mean_cluster_size":[float(np.mean(cs))],
            "median":[float(np.median(cs))],
            "p90":[float(np.percentile(cs,90))],
            "max":[int(np.max(cs))]
        })
        dup_stats.to_csv(OUTDIR / "dup_stats.csv", index=False)

        # 直方图（簇大小）
        plt.figure(figsize=(6.5,4))
        plt.hist(cs, bins=50)
        plt.xlabel("Cluster size (duplicates per unique sentence)")
        plt.ylabel("Count")
        plt.title("Duplicate cluster size distribution")
        plt.tight_layout()
        plt.savefig(OUTDIR / "dup_cluster_hist.png", dpi=180)
        plt.close()

# 5) 若去重后有 tokens，补充 top 词&短语、句长直方图
all_tokens = [t for ts in tokens_ded for t in ts]
ctr = Counter(all_tokens)
phr = [(w,c) for w,c in ctr.items() if "_" in w]
uni = [(w,c) for w,c in ctr.items() if "_" not in w]
top_phr = pd.DataFrame(sorted(phr, key=lambda x:(-x[1],x[0]))[:30], columns=["token","count"])
top_uni = pd.DataFrame(sorted(uni, key=lambda x:(-x[1],x[0]))[:30], columns=["token","count"])
top_phr.to_csv(OUTDIR / "dedup_top_phrases.csv", index=False)
top_uni.to_csv(OUTDIR / "dedup_top_unigrams.csv", index=False)

lens = [len(ts) for ts in tokens_ded]
plt.figure(figsize=(7,4))
plt.hist(lens, bins=50)
plt.xlabel("Tokens per sentence"); plt.ylabel("Count")
plt.title("Sentence length distribution (dedup)")
plt.tight_layout()
plt.savefig(OUTDIR / "dedup_token_len_hist.png", dpi=180)
plt.close()

# 6) 控制台打印（给你一眼确认用）
print("===== ORIGINAL (BEFORE DEDUP) =====")
print(stats_full_df.to_string(index=False))
print(f"\nExact unique sentences: {n_unique} / {n_total}  (dup_rate={dup_rate:.2%})")

print("\n===== DEDUP (LDA INPUT) =====")
print(stats_ded_df.to_string(index=False))

print("\n===== COMPARE =====")
print(compare_df.to_string(index=False))

print(f"\n[ok] Files saved -> {OUTDIR.resolve()}")