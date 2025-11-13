# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

# ========= 配置 =========
BASE = Path("/root/autodl-fs/lda_sentence_k37_run")
FILE_LDA_TOP = BASE / "topics_top_words.csv"
FILE_TFIDF   = BASE / "topic_top_phrases_mined.csv"
FILE_REP     = BASE / "topic_representative_sentences.csv"

OUT_FILE = BASE / "topic_labeling_table_clean.csv"

# ========= 要剔除的无意义词组 =========
stop_phrases = {
    "limit_experience",
    "acquire_business",
    "chief_executive_officer",
    "result_material",
    "adversely_affect_value_share",
    "united_states"
}

def clean_and_topn(series, topn=10):
    """清洗并取前topn个词/短语"""
    terms = [str(x).strip().lower() for x in series.head(50)]  # 多取一些，避免过滤后不够
    terms = [t for t in terms if t and t not in stop_phrases]
    return ", ".join(terms[:topn])

# ========= 读数据 =========
lda_df = pd.read_csv(FILE_LDA_TOP)
tfidf_df = pd.read_csv(FILE_TFIDF)
rep_df = pd.read_csv(FILE_REP)

# 1) 整理 LDA top words 前10个（过滤无意义词）
if "word" in lda_df.columns:
    word_col = "word"
elif "term" in lda_df.columns:
    word_col = "term"
elif "token" in lda_df.columns:
    word_col = "token"
else:
    raise ValueError("LDA 文件缺少词列")

lda_top = (lda_df.groupby("topic_id")[word_col]
           .apply(lambda x: clean_and_topn(x, topn=10))
           .reset_index(name="lda_top10"))

# 2) 整理 TF-IDF 短语 前10个（过滤无意义词）
if "phrase" in tfidf_df.columns:
    ph_col = "phrase"
elif "term" in tfidf_df.columns:
    ph_col = "term"
elif "token" in tfidf_df.columns:
    ph_col = "token"
elif "word" in tfidf_df.columns:
    ph_col = "word"
else:
    raise ValueError("TF-IDF 文件缺少短语列")

tfidf_top = (tfidf_df.groupby("topic_id")[ph_col]
             .apply(lambda x: clean_and_topn(x, topn=10))
             .reset_index(name="tfidf_top10"))

# 3) 整理代表性句子 只取1条
rep_top = (rep_df.groupby("topic_id")["text"]
           .apply(lambda x: x.head(1).iloc[0] if len(x) > 0 else "")
           .reset_index(name="rep_sentence"))

# ========= 合并 =========
merged = lda_top.merge(tfidf_top, on="topic_id", how="outer")
merged = merged.merge(rep_top, on="topic_id", how="outer")

# ========= 保存 =========
merged.to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f"[SAVE] {OUT_FILE}")