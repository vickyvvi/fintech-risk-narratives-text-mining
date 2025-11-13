#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_phrasers_and_filter.py
- 复用已训练的 bigram / trigram phraser，对 tokens 应用短语合成
- 先解短语(unphrase) 再去停用短语(stopphrases)
- 产出最终文件：/content/sentlda_stage/sentlda_phrases.parquet
  （含 tokens_bi, tokens_tri, tokens_final, tokens_final_str）
- 另外输出短语频次：sentlda_phrase_freq.csv / sentlda_top100_phrases.csv
"""

import os
import argparse
from collections import Counter
import pandas as pd
from gensim.models.phrases import Phraser

def parse_args():
    ap = argparse.ArgumentParser(description="Apply existing bigram/trigram phrasers + unphrase + stopphrases")
    ap.add_argument("--input", default="/content/sentlda_stage/sentlda_tokens_fixed.parquet",
                    help="输入含 tokens 的 parquet")
    ap.add_argument("--bigram-phraser", default="/content/sentlda_stage/sentlda_bigram.phraser",
                    help="已训练的 bigram.phraser 路径")
    ap.add_argument("--trigram-phraser", default="/content/sentlda_stage/sentlda_trigram.phraser",
                    help="已训练的 trigram.phraser 路径")
    ap.add_argument("--unphrase", default="/content/sentlda_stage/unphrase.txt",
                    help="解短语清单（每行一个短语；空格/短横线允许，会统一成下划线）")
    ap.add_argument("--stopphrases", default="/content/sentlda_stage/stop_phrases_v2.txt",
                    help="停用短语清单（每行一个短语；空格/短横线允许，会统一成下划线）")
    ap.add_argument("--output", default="/content/sentlda_stage/sentlda_phrases.parquet",
                    help="输出 parquet（固定文件名建议用 sentlda_phrases.parquet）")
    args, _ = ap.parse_known_args()
    return args

def load_phrases_file(path: str):
    """读取短语清单，清洗：strip -> lower -> 空格/短横线替换为下划线"""
    st = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                w = ln.strip().lower()
                if not w or w.startswith("#"):
                    continue
                w = w.replace(" ", "_").replace("-", "_")
                st.add(w)
    return st

def unphrase_tokens(tokens, blacklist):
    """将黑名单中的短语 token 拆回原词；其他 token 保持不变。"""
    out = []
    for t in tokens:
        tl = str(t).lower()
        if "_" in t and tl in blacklist:
            out.extend(t.split("_"))
        else:
            out.append(t)
    return out

def phrase_frequencies(docs):
    cnt = Counter()
    for ts in docs:
        for t in ts:
            if "_" in str(t):
                cnt[t] += 1
    df = pd.DataFrame({"phrase": list(cnt.keys()), "count": list(cnt.values())})
    return df.sort_values("count", ascending=False, kind="mergesort").reset_index(drop=True)

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    prefix = os.path.join(os.path.dirname(args.output), "sentlda")

    # 读 tokens
    df = pd.read_parquet(args.input)
    assert "tokens" in df.columns, "输入必须包含 'tokens' 列"
    docs = df["tokens"].tolist()
    print(f"[info] rows={len(df)}")

    # 加载 phrasers
    assert os.path.exists(args.bigram_phraser), f"bigram phraser not found: {args.bigram_phraser}"
    assert os.path.exists(args.trigram_phraser), f"trigram phraser not found: {args.trigram_phraser}"
    big_phraser = Phraser.load(args.bigram_phraser)
    tri_phraser = Phraser.load(args.trigram_phraser)
    print("[ok] loaded phrasers")

    # 应用 bigram -> trigram
    docs_bi  = [big_phraser[ts] for ts in docs]
    docs_tri = [tri_phraser[ts] for ts in docs_bi]

    # 统计短语频次（供人工审查）
    ph_freq = phrase_frequencies(docs_tri)
    ph_freq.to_csv(prefix + "_phrase_freq.csv", index=False, encoding="utf-8")
    ph_freq.head(100).to_csv(prefix + "_top100_phrases.csv", index=False, encoding="utf-8")
    print(f"[ok] saved phrase freq: {prefix}_phrase_freq.csv")
    print(f"[ok] saved top100:      {prefix}_top100_phrases.csv")

    # 读取 & 清洗 unphrase / stopphrases
    unphrase_set = load_phrases_file(args.unphrase)
    stopphr = load_phrases_file(args.stopphrases)
    print(f"[cfg] unphrase: {len(unphrase_set)} phrases, stopphrases: {len(stopphr)} phrases")

    # 先解短语，再去停用短语
    if unphrase_set:
        docs_tri = [unphrase_tokens(ts, unphrase_set) for ts in docs_tri]
    docs_final = [[t for t in ts if str(t).lower() not in stopphr] for ts in docs_tri]

    # 输出最终 parquet
    out_df = df.copy()
    out_df["tokens_bi"] = docs_bi
    out_df["tokens_tri"] = docs_tri
    out_df["tokens_final"] = docs_final
    out_df["tokens_final_str"] = [" ".join(map(str, ts)) for ts in docs_final]
    out_df.to_parquet(args.output, index=False)
    print(f"[ok] saved final: {args.output}")

    print("------ SUMMARY ------")
    print(f"rows: {len(out_df)}")
    print(f"phrases (unique) in tri stage: {ph_freq.shape[0]}")
    if len(docs_final) > 0:
        avg_len = sum(len(ts) for ts in docs_final) / len(docs_final)
        print(f"avg tokens_final length: {avg_len:.2f}")

if __name__ == "__main__":
    main()

import pandas as pd
from collections import Counter

df = pd.read_parquet("/content/sentlda_stage/sentlda_phrases.parquet")

print("rows:", len(df))
print("sample tokens_final:")
print(df["tokens_final"].head(3).tolist())

# 看看短语（含下划线）的保留Top 30，确认没把有用短语误删
phr_tokens = (t for ts in df["tokens_final"] for t in ts if "_" in t)
cnt = Counter(phr_tokens)
print("\nTOP 200 phrases in tokens_final:")
for w,c in cnt.most_common(30):
    print(f"{w:30s} {c}")

import pandas as pd

# 读取短语频次表
ph_freq = pd.read_csv("/content/sentlda_stage/sentlda_phrase_freq.csv")
print("总共的短语数量:", len(ph_freq))

# 看尾部几个（低频短语）
print(ph_freq.tail(10))

import pandas as pd
from collections import Counter

# 读文件
df = pd.read_parquet("/content/sentlda_stage/sentlda_phrases.parquet")

# 抽样看几行
print("Sample tokens_final (前3行):")
for row in df["tokens_final"].head(3):
    print(row)

# 统计 unigram vs ngram
all_tokens = [t for ts in df["tokens_final"] for t in ts]
unigrams = [t for t in all_tokens if "_" not in t]
ngrams   = [t for t in all_tokens if "_" in t]

print("\n===== STATS =====")
print("总token数:", len(all_tokens))
print("单词数 (unigram):", len(unigrams))
print("词组数 (bigram/trigram):", len(ngrams))
print("词组比例: {:.2%}".format(len(ngrams)/len(all_tokens)))

# 看高频前20（混合情况）
cnt = Counter(all_tokens)
print("\nTop 20 tokens overall:")
for w,c in cnt.most_common(20):
    print(f"{w:25s} {c}")

# 高频前10词组
print("\nTop 10 phrases (含下划线):")
for w,c in [(w,c) for w,c in cnt.most_common() if "_" in w][:10]:
    print(f"{w:25s} {c}")

# 高频前10单词
print("\nTop 10 unigrams:")
for w,c in [(w,c) for w,c in cnt.most_common() if "_" not in w][:10]:
    print(f"{w:25s} {c}")