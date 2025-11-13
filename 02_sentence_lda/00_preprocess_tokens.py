#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step24_clean_tokens.py
步骤 2–4：
  2) 分词（可把数字标准化为 <NUM>）
  3) 停用词去除（spaCy 自带 + 你的 final_stopwords_v8.txt）
  4) 词形还原（优先 spaCy，回退 NLTK WordNet）

输入（默认）:
  --input /content/risk_sentence_enriched_dedup.parquet
  --text-col sentence_sub
  --extra-stopwords /mnt/data/final_stopwords_v8.txt

输出：
  {output_prefix}_tokens.parquet  ：tokens(list) + tokens_str
  同目录 CSV 兜底（若 parquet 引擎缺失）
"""

import os, re, argparse
import pandas as pd
from typing import List, Set

# -------- args --------
def parse_args():
    ap = argparse.ArgumentParser(description="Tokenize + stopword removal + lemmatization for sentence-level corpus.")
    ap.add_argument("--input", default="/content/risk_sentence_enriched_dedup.parquet")
    ap.add_argument("--text-col", default="sentence_sub")
    ap.add_argument("--output-prefix", default="/content/sentlda_stage/sentlda")
    ap.add_argument("--extra-stopwords", default="/mnt/data/final_stopwords_v8.txt", help="自建停用词，每行一个词")
    ap.add_argument("--keep-numbers", action="store_true", help="保留原数字（默认替换为 <NUM>）")
    ap.add_argument("--lower", action="store_true", help="转小写（建议开）")
    ap.add_argument("--sample-cap", type=int, default=0, help="仅处理前 N 行（调试用）")
    args, _ = ap.parse_known_args()
    return args

# -------- IO --------
def read_table_auto(path: str) -> pd.DataFrame:
    assert os.path.exists(path), f"Input not found: {path}"
    if path.endswith(".parquet"):
        for eng in ("pyarrow","fastparquet"):
            try: return pd.read_parquet(path, engine=eng)
            except Exception: pass
    # fallback csv/tsv
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)

def write_parquet_or_csv(df: pd.DataFrame, out_path_no_ext: str):
    pqt = out_path_no_ext + "_tokens.parquet"
    try:
        for eng in ("pyarrow","fastparquet"):
            try:
                df.to_parquet(pqt, engine=eng, index=False)
                print(f"[ok] saved {pqt} (engine={eng})")
                return
            except Exception:
                pass
        raise RuntimeError("no parquet engine")
    except Exception:
        csvp = out_path_no_ext + "_tokens.csv"
        df.to_csv(csvp, index=False, encoding="utf-8")
        print(f"[warn] parquet unavailable; saved CSV {csvp}")

# -------- tokenizer --------
NUM_RE = re.compile(r"\d+(?:[,\.\d]*\d)?")

def basic_tokenize(s: str, keep_numbers: bool, to_lower: bool) -> List[str]:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = re.sub(r"<[^>]+>", " ", s)  # 去 HTML
    if to_lower:
        s = s.lower()
    if keep_numbers:
        s = re.sub(r"(\d),(?=\d{3}\b)", r"\1", s)  # 1,000 -> 1000
    else:
        s = NUM_RE.sub("<NUM>", s)
    # 只保留字母词和 <NUM>
    return re.findall(r"<NUM>|[A-Za-z]+(?:'[A-Za-z]+)?", s)

# -------- lemmatizer --------
class Lemmatizer:
    def __init__(self):
        self.mode = None
        self.nlp = None
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser","ner","textcat"])
            except Exception:
                from spacy.cli import download
                download("en_core_web_sm")
                import spacy as _sp
                self.nlp = _sp.load("en_core_web_sm", disable=["parser","ner","textcat"])
            self.mode = "spacy"
        except Exception:
            self.mode = "nltk"
            import nltk
            try:
                nltk.data.find("corpora/wordnet")
            except Exception:
                nltk.download("wordnet"); nltk.download("omw-1.4")
            from nltk.stem import WordNetLemmatizer
            self.wl = WordNetLemmatizer()

    def __call__(self, tokens: List[str]) -> List[str]:
        if not tokens: return tokens
        if self.mode == "spacy":
            doc = self.nlp(" ".join(tokens))
            return [t.lemma_ if t.lemma_ else t.text for t in doc]
        else:
            return [self.wl.lemmatize(t) for t in tokens]

# -------- stopwords --------
def load_extra_stopwords(path: str) -> Set[str]:
    stop = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                w = ln.strip().lower()
                if w and not w.startswith("#"):
                    stop.add(w)
    return stop

def build_stopwords(extra_path: str) -> Set[str]:
    sw = set()
    # spaCy 英文停用词
    try:
        import spacy
        from spacy.lang.en.stop_words import STOP_WORDS as SPACY_SW
        sw |= {w.lower() for w in SPACY_SW}
    except Exception:
        pass
    # 自建停用词
    sw |= load_extra_stopwords(extra_path)
    return sw

# -------- main --------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    df = read_table_auto(args.input)
    assert args.text_col in df.columns, f"Text column '{args.text_col}' not found."
    if args.sample_cap and len(df) > args.sample_cap:
        df = df.iloc[:args.sample_cap].copy()
        print(f"[info] sample_cap active: rows={len(df)}")

    texts = df[args.text_col].fillna("").astype(str).tolist()

    # 2) tokenize
    toks = [basic_tokenize(s, keep_numbers=args.keep_numbers, to_lower=(args.lower or True)) for s in texts]

    # 4) lemmatize
    lem = Lemmatizer()
    toks = [lem(ts) for ts in toks]

    # 3) stopwords remove（spaCy + 你的 final_stopwords）
    stopwords = build_stopwords(args.extra_stopwords)

    def filt(ts: List[str]) -> List[str]:
        out = []
        for t in ts:
            if t in {"<NUM>", "num"}:  # 统一丢掉数字标记
                continue
            tt = t.lower()
            if tt in stopwords:  # 停用词过滤
                continue
            if len(tt) == 1:     # 单字符过滤
                continue
            out.append(tt)
        return out

    toks_clean = [filt(ts) for ts in toks]

    out_df = df.copy()
    out_df["tokens"] = toks_clean
    out_df["tokens_str"] = [" ".join(ts) for ts in toks_clean]

    base = args.output_prefix
    write_parquet_or_csv(out_df, base)

    print("------ SUMMARY (step 2–4) ------")
    print(f"rows: {len(out_df)}")
    print(f"saved: {base}_tokens.parquet (or CSV fallback)")

if __name__ == "__main__":
    main()

import pandas as pd

df = pd.read_parquet("/content/sentlda_stage/sentlda_tokens.parquet")

# 打印前 5 行（只挑常用几列）
print(df[["sentence_sub", "tokens", "tokens_str"]].head(5).to_string(index=False))

# 统计信息
all_tokens = [t for ts in df["tokens"] for t in ts]   # 展开所有词
vocab_size = len(set(all_tokens))
avg_len = df["tokens"].map(len).mean()

print("\n===== STATS =====")
print(f"总句子数: {len(df)}")
print(f"词表大小: {vocab_size}")
print(f"平均句长: {avg_len:.2f} tokens")

from collections import Counter

all_tokens = [t for ts in df["tokens"] for t in ts]
counter = Counter(all_tokens)

print("===== TOP 20 TOKENS =====")
for w, c in counter.most_common(20):
    print(f"{w:15s} {c}")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_tokens_from_str.py
从 tokens_str 列重建 tokens 列（list of str），同时去掉 <NUM>/num。

输入:
  --input /content/sentlda_stage/sentlda_tokens.parquet
  --output-prefix /content/sentlda_stage/sentlda

输出:
  {output_prefix}_tokens_fixed.parquet
"""

import os, argparse, pandas as pd
from collections import Counter

def parse_args():
    ap = argparse.ArgumentParser(description="Fix tokens column from tokens_str, drop <NUM>/num.")
    ap.add_argument("--input", default="/content/sentlda_stage/sentlda_tokens.parquet")
    ap.add_argument("--output-prefix", default="/content/sentlda_stage/sentlda")
    args, _ = ap.parse_known_args()
    return args

def main():
    args = parse_args()
    df = pd.read_parquet(args.input)

    assert "tokens_str" in df.columns, "输入文件必须包含 'tokens_str' 列"

    # 用 tokens_str 重建 tokens
    df["tokens"] = df["tokens_str"].fillna("").apply(
        lambda x: [t for t in str(x).split() if t.lower() not in {"<num>", "num"}]
    )

    # 重新拼接
    df["tokens_str"] = [" ".join(ts) for ts in df["tokens"]]

    # 保存
    out_path = args.output_prefix + "_tokens_fixed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[ok] saved: {out_path}")

    # 打印统计
    all_tokens = [t for ts in df["tokens"] for t in ts]
    counter = Counter(all_tokens)

    print("===== TOP 20 TOKENS =====")
    for w, c in counter.most_common(20):
        print(f"{w:15s} {c}")

    print("\n===== STATS =====")
    print(f"总句子数: {len(df)}")
    print(f"词表大小: {len(set(all_tokens))}")
    print(f"平均句长: {sum(len(ts) for ts in df['tokens'])/len(df):.2f} tokens")

if __name__ == "__main__":
    main()