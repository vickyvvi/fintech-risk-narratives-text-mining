!pip install gensim

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_phrase_candidates.py
Step A: 短语挖掘（bigram -> trigram），输出候选短语频次表（不做停用短语过滤）。

输出：
  {output_prefix}_phrases_raw.parquet : 含 tokens_bi, tokens_tri
  {output_prefix}_phrase_freq.csv     : 所有 bigram/trigram 频次
  {output_prefix}_top100_phrases.csv  : 前100 高频短语
  {output_prefix}_bigram.phraser / _trigram.phraser : 训练好的模型
"""

import os, argparse, pandas as pd
from gensim.models.phrases import Phrases, Phraser
from collections import Counter

def parse_args():
    ap = argparse.ArgumentParser(description="Phrase mining (bigram -> trigram), output candidates only")
    ap.add_argument("--input", default="/content/sentlda_stage/sentlda_tokens_fixed.parquet")
    ap.add_argument("--output-prefix", default="/content/sentlda_stage/sentlda")
    ap.add_argument("--bigram-min-count", type=int, default=10)
    ap.add_argument("--bigram-threshold", type=float, default=0.3,
                    help="当 scoring='npmi' 时应在 [-1,1]，常用 0.1~0.5")
    ap.add_argument("--trigram-min-count", type=int, default=5)
    ap.add_argument("--trigram-threshold", type=float, default=0.2,
                    help="当 scoring='npmi' 时应在 [-1,1]，常用 0.05~0.3")
    ap.add_argument("--scoring", default="npmi", choices=["npmi","default"],
                    help="npmi 更稳健；若选 default，阈值可用 10/8 等较大数")
    args, _ = ap.parse_known_args()
    return args

def _safe_threshold(thr: float, scoring: str, fallback: float) -> float:
    if scoring == "npmi":
        # npmi 需要 [-1,1]；若传了 >1，自动回落到 fallback（默认 0.3/0.2）
        if thr < -1.0: return -1.0
        if thr > 1.0:  return fallback
        return thr
    # default 允许较大阈值
    return thr

def phrase_frequencies(docs):
    cnt = Counter()
    for ts in docs:
        for t in ts:
            if "_" in t:  # 只统计短语
                cnt[t]+=1
    return pd.DataFrame({"phrase":list(cnt.keys()),"count":list(cnt.values())}).sort_values("count",ascending=False)

def main():
    args = parse_args()
    df = pd.read_parquet(args.input)
    assert "tokens" in df.columns, "输入必须包含 'tokens' 列"
    docs = df["tokens"].tolist()

    # 规范化阈值
    big_thr = _safe_threshold(args.bigram_threshold, args.scoring, fallback=0.3)
    tri_thr = _safe_threshold(args.trigram_threshold, args.scoring, fallback=0.2)

    print(f"[cfg] scoring={args.scoring}, bigram(min_count={args.bigram_min_count}, threshold={big_thr}), "
          f"trigram(min_count={args.trigram_min_count}, threshold={tri_thr})")

    # bigram
    print("[stage] training bigram...")
    big = Phrases(docs, min_count=args.bigram_min_count, threshold=big_thr, scoring=args.scoring)
    big_phraser = Phraser(big)
    docs_bi = [big_phraser[ts] for ts in docs]

    # trigram
    print("[stage] training trigram...")
    tri = Phrases(docs_bi, min_count=args.trigram_min_count, threshold=tri_thr, scoring=args.scoring)
    tri_phraser = Phraser(tri)
    docs_tri = [tri_phraser[ts] for ts in docs_bi]

    # 短语频次
    ph_freq = phrase_frequencies(docs_tri)
    ph_freq.to_csv(args.output_prefix+"_phrase_freq.csv", index=False, encoding="utf-8")
    ph_freq.head(100).to_csv(args.output_prefix+"_top100_phrases.csv", index=False, encoding="utf-8")
    print(f"[ok] saved phrase freq: {args.output_prefix}_phrase_freq.csv")
    print(f"[ok] saved top100:      {args.output_prefix}_top100_phrases.csv")

    # 保存 phraser 模型
    big_phraser.save(args.output_prefix+"_bigram.phraser")
    tri_phraser.save(args.output_prefix+"_trigram.phraser")

    # 保存含 bi/tri 短语的 tokens（不做停用过滤）
    df["tokens_bi"] = docs_bi
    df["tokens_tri"] = docs_tri
    df.to_parquet(args.output_prefix+"_phrases_raw.parquet", index=False)
    print(f"[ok] saved tokens with phrases: {args.output_prefix}_phrases_raw.parquet")

    print("------ SUMMARY ------")
    print(f"rows: {len(df)}")
    print(f"短语总数: {len(ph_freq)}")

if __name__=="__main__":
    main()

