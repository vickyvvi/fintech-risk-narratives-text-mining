# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from collections import Counter
from wordcloud import WordCloud

# ========= 配置 =========
RUN_DIR      = Path("/root/autodl-fs/lda_sentence_k37_run")   # 你LDA输出目录
DOC_TOPICS   = RUN_DIR / "doc_topics.parquet"
ORIG_TABLE   = Path("/root/autodl-fs/sentlda_final_v2_noempty.parquet")  # 原始全表

# 可能的“分词/文本”列候选（按优先级从高到低）
TEXT_COL_CAND = [
    "tokens_final", "tokens_tri", "tokens_bi", "tokens", "tokens_final_str",
    "tokens_str", "sentence_sub", "sentence_raw"
]
# 原文列候选
RAW_COL_CAND = [
    "sentence_raw","sentence_original","raw_text","sentence","text_raw","orig_text"
]
# 元数据候选（存在则带）
META_CAND    = [
    "doc_id","sent_id","Company","Ticker","Year","FilingType",
    "Sub-Sector","SubSector","Industry","Sector",
    "Headquarters location","HQ","Country","Region"
]

K_TOP_DOCS_FOR_PHRASES = 2000   # 每主题训练短语器时使用的Top句子数
TOP_SAMPLE = 50                 # 每主题样本Top/Bottom各50
PHRASE_TOPN = 100               # 每主题导出Top短语个数
BIGRAM_MIN_COUNT = 10           # Phrases超参
BIGRAM_THRESHOLD = 10.0
TRIGRAM_THRESHOLD = 8.0

def pick_first_existing(candidates, df):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def as_tokens_series(series):
    """
    统一把一列（可能是 list，也可能是空格分词后的 str）转成 List[str]。
    """
    vals = []
    for x in series.fillna(""):
        if isinstance(x, (list, tuple, np.ndarray)):
            vals.append([str(t).strip() for t in list(x) if str(t).strip()])
        else:
            s = str(x).strip()
            vals.append([w for w in s.split() if w.strip()])
    return vals

def smart_merge(doc_topics, orig):
    """
    优先用 (doc_id, sent_id) 合并；无则退回 orig_row；再无就并排拼（最差兜底）。
    右表只带非 key 列，避免 key 列重复。
    """
    keys = None
    if {"doc_id","sent_id"}.issubset(doc_topics.columns) and {"doc_id","sent_id"}.issubset(orig.columns):
        keys = ["doc_id","sent_id"]
    elif "orig_row" in doc_topics.columns and "orig_row" in orig.columns:
        keys = ["orig_row"]

    raw_col  = pick_first_existing(RAW_COL_CAND, orig)
    text_col = pick_first_existing(TEXT_COL_CAND, orig)
    meta_use = [c for c in META_CAND if c in orig.columns]

    plus_cols = [c for c in [text_col, raw_col] if c and c in orig.columns]

    if keys:
        right_cols = [c for c in [*meta_use, *plus_cols] if c not in keys]
        right = orig[[*keys, *right_cols]].drop_duplicates(keys)
        merged = doc_topics.merge(right, on=keys, how="left")
    else:
        # 没有键，只能拼上能拿到的列（行可能对不齐，仅兜底用）
        merged = pd.concat(
            [doc_topics.reset_index(drop=True),
             orig[[*(meta_use + plus_cols)]].reset_index(drop=True)],
            axis=1
        )

    # 返回：合并后的表、选到的原文列、选到的文本列、最终可用的元数据列（在 merged 里存在的）
    meta_in_merged = [c for c in meta_use if c in merged.columns]
    return merged, raw_col, text_col, meta_in_merged

def train_phraser_on_topic(top_texts_tokens):
    """
    在该主题Top句子的token上，训练bigram->trigram短语器，并返回短语化后的tokens列表。
    """
    if len(top_texts_tokens) == 0:
        return top_texts_tokens

    bigram = Phrases(top_texts_tokens, min_count=BIGRAM_MIN_COUNT, threshold=BIGRAM_THRESHOLD)
    bigram_phraser = Phraser(bigram)
    bi_corpus = [bigram_phraser[doc] for doc in top_texts_tokens]

    trigram = Phrases(bi_corpus, min_count=BIGRAM_MIN_COUNT, threshold=TRIGRAM_THRESHOLD)
    trigram_phraser = Phraser(trigram)
    tri_corpus = [trigram_phraser[doc] for doc in bi_corpus]

    return tri_corpus

def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "wordclouds").mkdir(exist_ok=True)

    # 1) 读取 doc_topics 与 原始表
    dt = pd.read_parquet(DOC_TOPICS)
    orig = pd.read_parquet(ORIG_TABLE)
    # 极少数 parquet 会带重复列名，这里去重一次更保险
    orig = orig.loc[:, ~orig.columns.duplicated()].copy()

    # 2) 合并（补齐 raw 与元数据）
    merged, raw_col_from_orig, text_col_from_orig, meta_cols = smart_merge(dt, orig)

    # >>> 新增：合并后在 merged 里重选一下文本/原文列（优先 merged，次选 orig）
    TEXT_COL_CAND = [
        "tokens_final", "tokens_tri", "tokens_bi", "tokens", "tokens_final_str",
        "tokens_str", "sentence_sub", "sentence_raw"
    ]
    RAW_COL_CAND = [
        "sentence_raw","sentence_original","raw_text","sentence","text_raw","orig_text"
    ]
    text_col_m = pick_first_existing(TEXT_COL_CAND, merged)
    raw_col_m  = pick_first_existing(RAW_COL_CAND,  merged)

    # 如果 merged 里都没有，再考虑 orig 的列名（备用）
    if text_col_m is None:
        text_col_m = pick_first_existing(TEXT_COL_CAND, orig)
    if raw_col_m is None:
        raw_col_m = pick_first_existing(RAW_COL_CAND,  orig)

    # 3) Top/Bottom 样本（含 raw & 元数据，全部“存在才取”）
    theta_cols = [c for c in merged.columns if c.startswith("theta_")]
    if not theta_cols:
        raise ValueError("doc_topics.parquet 里没有 theta_* 列。")

    K = len(theta_cols)
    samples = []
    for k in range(K):
        col = f"theta_{k}"
        scores = merged[col].to_numpy()

        # Top
        top_idx = np.argsort(-scores)[:TOP_SAMPLE]
        # Bottom（优先 >0）
        nz = np.where(scores > 0)[0]
        pool = nz if len(nz) >= TOP_SAMPLE else np.arange(len(scores))
        bot_idx = pool[np.argsort(scores[pool])[:TOP_SAMPLE]]

        for which, arr in [("top", top_idx), ("bottom", bot_idx)]:
            for rank, i in enumerate(arr, 1):
                row = {"topic_id": k, "which": which, "rank": rank,
                       "prob": float(scores[i])}
                if "dominant_topic" in merged.columns:
                    row["dominant_topic"] = int(merged.loc[i, "dominant_topic"])
                if "dominant_prob" in merged.columns:
                    row["dominant_prob"] = float(merged.loc[i, "dominant_prob"])
                if "orig_row" in merged.columns:
                    row["orig_row"] = int(merged.loc[i, "orig_row"])

                # 元数据（只取 merged 里存在的）
                for c in meta_cols:
                    if c in merged.columns:
                        row[c] = merged.loc[i, c]

                # 文本：优先 merged 的列，其次才用 orig 的同名列兜底
                if text_col_m and text_col_m in merged.columns:
                    row[text_col_m] = merged.loc[i, text_col_m]
                elif raw_col_m and raw_col_m in merged.columns:
                    row[raw_col_m] = merged.loc[i, raw_col_m]
                samples.append(row)

    samples_df = pd.DataFrame(samples)
    out_samples = RUN_DIR / "samples_top_bottom_with_raw.csv"
    samples_df.to_csv(out_samples, index=False)
    print("[SAVE]", out_samples)

    # 4) 准备 all_tokens 供短语挖掘：
    #  优先：merged 中的 tokens/text 列
    #  其次：如果 merged 没有，但 merged 有 orig_row，则回到 orig 用 orig_row 映射拿文本
    def as_tokens_series(series):
        vals = []
        for x in series.fillna(""):
            if isinstance(x, (list, tuple, np.ndarray)):
                vals.append([str(t).strip() for t in list(x) if str(t).strip()])
            else:
                s = str(x).strip()
                vals.append([w for w in s.split() if w.strip()])
        return vals

    all_tokens = None
    if text_col_m and text_col_m in merged.columns:
        all_tokens = as_tokens_series(merged[text_col_m])
    elif raw_col_m and raw_col_m in merged.columns:
        all_tokens = as_tokens_series(merged[raw_col_m])
    elif "orig_row" in merged.columns:
        # 回到 orig 拿文本，并对齐 merged 顺序
        # 先在 orig 里找到一个可用的文本列（tokens_*优先，其次 raw_*）
        text_col_o = pick_first_existing(TEXT_COL_CAND, orig)
        raw_col_o  = pick_first_existing(RAW_COL_CAND,  orig)
        use_col = text_col_o or raw_col_o
        if use_col is None:
            raise ValueError("原表里也没有可用的文本/原文列，无法构造 tokens。")
        # 建映射：orig_row -> 文本
        if "orig_row" not in orig.columns:
            raise ValueError("无法从 orig 回链，因为 orig 缺少 orig_row。")
        mapper = orig.set_index("orig_row")[use_col]
        series_aligned = merged["orig_row"].map(mapper)
        all_tokens = as_tokens_series(series_aligned)
    else:
        raise ValueError("既没有文本列也没有 raw 列，且无法通过 orig_row 回链文本。")

    # === 下面保持你的短语训练与词云逻辑（稍做函数名适配） ===
    def train_phraser_on_topic(top_texts_tokens):
        from gensim.models.phrases import Phrases, Phraser
        if len(top_texts_tokens) == 0:
            return top_texts_tokens
        bigram = Phrases(top_texts_tokens, min_count=BIGRAM_MIN_COUNT, threshold=BIGRAM_THRESHOLD)
        bigram_phraser = Phraser(bigram)
        bi_corpus = [bigram_phraser[doc] for doc in top_texts_tokens]
        trigram = Phrases(bi_corpus, min_count=BIGRAM_MIN_COUNT, threshold=TRIGRAM_THRESHOLD)
        trigram_phraser = Phraser(trigram)
        tri_corpus = [trigram_phraser[doc] for doc in bi_corpus]
        return tri_corpus

    phrase_rows = []
    for k in range(K):
        col = f"theta_{k}"
        scores = merged[col].to_numpy()

        order = np.argsort(-scores)[:K_TOP_DOCS_FOR_PHRASES]
        top_tokens = [all_tokens[i] for i in order]

        tri_corpus = train_phraser_on_topic(top_tokens)

        # 统计短语（下划线视为短语）
        cnt = Counter()
        for doc in tri_corpus:
            for tok in doc:
                if "_" in tok:
                    cnt[tok] += 1

        if len(cnt) < 20:
            # 兜底：补 unigram
            unis = Counter()
            for doc in tri_corpus:
                for tok in doc:
                    if "_" not in tok:
                        unis[tok] += 1
            for w, c in unis.most_common(50):
                cnt[w] = c

        # 保存短语Top表
        for rank, (ph, c) in enumerate(cnt.most_common(PHRASE_TOPN), 1):
            phrase_rows.append({"topic_id": k, "rank": rank, "phrase": ph, "count": int(c)})

        # 词云
        if cnt:
            freq = {ph.replace("_", "\u00A0"): c for ph, c in cnt.items()}
            wc = WordCloud(width=1600, height=900, background_color="white",
                           max_words=200, collocations=False)
            out_png = RUN_DIR / f"wordclouds/topic_{k:02d}_phrases.png"
            wc.generate_from_frequencies(freq)
            wc.to_file(out_png.as_posix())

    phrase_csv = RUN_DIR / "topic_top_phrases_mined.csv"
    pd.DataFrame(phrase_rows).to_csv(phrase_csv, index=False)
    print("[SAVE]", phrase_csv)
    print("[DONE] 词云输出目录：", RUN_DIR / "wordclouds")


if __name__ == "__main__":
    main()
