# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
一次性跑 句子/子句级 LDA（K=27）并导出所有常用成果（无需命令行参数）：
- 固定词典（gensim Dictionary）
- 主题 top 词（CSV）
- 文档-主题分布（Parquet）：theta_*、dominant_topic、dominant_prob
- 每个主题 Top50 / Bottom50 代表句（CSV）
- 主题摘要（CSV）
- 训练日志与早停
- 保留 orig_row 以映射回输入表；输入表其它元数据（如 Company/Year 等）也会被一并保留

依赖：pandas, numpy, tqdm, tomotopy, gensim
"""

import os, math, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import tomotopy as tp
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


IN_PARQUET  = "/root/autodl-fs/sentlda_final_v2_noempty.parquet"  # 去重后的子句/句子表
TEXT_COL    = "tokens_final"  # 已空格分词的文本列
OUT_DIR     = "/root/autodl-fs/lda_sentence_k37_run"  # 输出目录

K                 = 2
SEED              = 42
ITERS_MAX         = 1000          # 最大训练步数（每 50 为一小步）
EARLY_PATIENCE    = 150           # 早停耐心（单位=迭代次数）
TOPN_TOPIC_WORDS  = 30            # 每个主题导出前多少个词
SAMPLE_TOP        = 50            # 每主题抽样 TopN 文档
SAMPLE_BOTTOM     = 50            # 每主题抽样 BottomN 文档（优先从>0概率集合中取）

# 词典裁剪
DICT_NO_BELOW     = 20            # 至少在多少文档中出现
DICT_NO_ABOVE     = 0.4           # 出现在多少比例以上的文档就裁剪
DICT_KEEP_N       = 50000         # 词表上限

# =========================

def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取数据 + 保留 orig_row 以便回溯
    df = pd.read_parquet(IN_PARQUET)
    if TEXT_COL not in df.columns:
        raise ValueError(f"找不到文本列 {TEXT_COL}；可用列示例：{list(df.columns)[:20]}")
    df = df.reset_index(drop=True)
    df["orig_row"] = np.arange(len(df), dtype=np.int64)
    texts = df[TEXT_COL].astype(str).fillna("").tolist()
    print(f"[LOAD] rows={len(df):,} | text_col={TEXT_COL}")

    # 2) 句子/子句 -> tokens（已空格分词的列，直接 split）
    tokens = [s.split() for s in tqdm(texts, desc="Tokenize(split)")]
    # 构建固定词典
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=DICT_NO_BELOW, no_above=DICT_NO_ABOVE, keep_n=DICT_KEEP_N)
    dictionary.compactify()
    dictionary.save((out_dir / "lda_dict.dict").as_posix())
    print(f"[DICT] vocab_size={len(dictionary):,} -> {out_dir/'lda_dict.dict'}")

    # 仅保留词典中词；并过滤空文档
    vocab_set = set(dictionary.token2id.keys())
    tokens_filtered = [[w for w in doc if w in vocab_set] for doc in tokens]
    mask = [len(doc) > 0 for doc in tokens_filtered]
    tokens_filtered = [doc for doc, keep in zip(tokens_filtered, mask) if keep]
    df_used = df.loc[mask].reset_index(drop=True)   # 保留 orig_row + 元数据
    # 持久化，方便复现
    np.save(out_dir / "tokens_filtered.npy", np.array(tokens_filtered, dtype=object), allow_pickle=True)
    df_used.to_parquet(out_dir / "df_used.parquet", index=False)
    print(f"[DATA] docs_after_filter={len(tokens_filtered):,} | saved tokens & df_used")

    # 3) 训练 LDA（K=27，seed固定、带早停）
    lda = tp.LDAModel(k=K, tw=tp.TermWeight.IDF, seed=SEED)
    for doc in tokens_filtered:
        lda.add_doc(doc)

    best_llpw  = -math.inf
    best_step  = -1
    history_ll = []

    steps = math.ceil(ITERS_MAX / 50)
    for step in range(steps):
        lda.train(50)
        llpw = lda.ll_per_word  # log-likelihood per word
        history_ll.append(float(llpw))
        # 记录最优
        if llpw > best_llpw + 1e-9:
            best_llpw = llpw
            best_step = step
        # 早停
        if (step - best_step) * 50 >= EARLY_PATIENCE:
            print(f"[EARLY STOP] step={step} | best_step={best_step} | best_llpw={best_llpw:.6f}")
            break
        if (step + 1) % 2 == 0:
            print(f"[TRAIN] step={(step+1):3d}*50 | ll_per_word={llpw:.6f}")

    # 4) 指标保存：perplexity & coherence(c_v)
    perp = float(np.exp(-lda.ll_per_word)) if lda.ll_per_word is not None else np.nan
    topics_for_coh = [[w for w, _ in lda.get_topic_words(k, top_n=10)] for k in range(K)]
    cm = CoherenceModel(topics=topics_for_coh, texts=tokens_filtered, dictionary=dictionary, coherence='c_v')
    coh_cv = float(cm.get_coherence())
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "K": K,
            "seed": SEED,
            "iters_done": (step + 1) * 50,
            "ll_per_word": float(lda.ll_per_word),
            "perplexity": perp,
            "coherence_c_v": coh_cv,
            "history_ll_per_word": history_ll,
            "dict": {"no_below": DICT_NO_BELOW, "no_above": DICT_NO_ABOVE, "keep_n": DICT_KEEP_N}
        }, f, ensure_ascii=False, indent=2)
    print(f"[METRICS] K={K} | iters={(step+1)*50} | perp={perp:.2f} | c_v={coh_cv:.3f}")

    # 5) 导出主题 top 词
    rows = []
    for k in range(K):
        for r, (w, p) in enumerate(lda.get_topic_words(k, top_n=TOPN_TOPIC_WORDS), start=1):
            rows.append({"topic_id": k, "rank": r, "word": w, "prob": float(p)})
    pd.DataFrame(rows).to_csv(out_dir / "topics_top_words.csv", index=False)
    print("[SAVE] topics_top_words.csv")

    # 6) 文档-主题分布（theta）、主导主题（保留 orig_row 便于回溯）
    thetas = [doc.get_topic_dist() for doc in lda.docs]
    Theta = np.array(thetas, dtype=np.float32)  # (N_docs x K)
    dom_idx = Theta.argmax(axis=1)
    dom_val = Theta.max(axis=1)

    theta_cols = [f"theta_{i}" for i in range(K)]
    doc_out = df_used.copy()  # 含 orig_row + 你的元数据列
    for i, col in enumerate(theta_cols):
        doc_out[col] = Theta[:, i]
    doc_out["dominant_topic"] = dom_idx
    doc_out["dominant_prob"]  = dom_val
    doc_out.to_parquet(out_dir / "doc_topics.parquet", index=False)
    print(f"[SAVE] doc_topics.parquet shape={doc_out.shape}")

    # 7) 主题摘要（规模、均值概率、Top10词）
    dom_counts = pd.Series(dom_idx).value_counts().to_dict()
    summary = []
    for k in range(K):
        top10 = [w for w, _ in lda.get_topic_words(k, top_n=10)]
        summary.append({
            "topic_id": k,
            "dom_doc_count": int(dom_counts.get(k, 0)),
            "mean_theta": float(Theta[:, k].mean()),
            "top10_words": " ".join(top10)
        })
    pd.DataFrame(summary).sort_values("dom_doc_count", ascending=False)\
        .to_csv(out_dir / "topics_summary.csv", index=False)
    print("[SAVE] topics_summary.csv")

    # 8) 每主题 Top50 / Bottom50 样本（保留 orig_row + 元数据 + 文本）
    # 便于人工命名与检查边界样本
    meta_cols = [c for c in ["orig_row","doc_id","sent_id","Company","Year","FilingType",
                             "Headquarters location","Sub-Sector"] if c in df_used.columns]
    base_cols = meta_cols + [TEXT_COL]
    sample_rows = []
    for k in range(K):
        scores = Theta[:, k]
        # Top
        top_idx = np.argsort(-scores)[:SAMPLE_TOP]
        for rank, i_doc in enumerate(top_idx, start=1):
            rec = {"topic_id": k, "which": "top", "rank": rank, "prob": float(scores[i_doc])}
            for c in base_cols: rec[c] = df_used.iloc[i_doc][c]
            sample_rows.append(rec)
        # Bottom（优先从 >0 概率池中取）
        nz = np.where(scores > 0)[0]
        bot_pool = nz if len(nz) >= SAMPLE_BOTTOM else np.arange(len(scores))
        bot_idx = bot_pool[np.argsort(scores[bot_pool])[:SAMPLE_BOTTOM]]
        for rank, i_doc in enumerate(bot_idx, start=1):
            rec = {"topic_id": k, "which": "bottom", "rank": rank, "prob": float(scores[i_doc])}
            for c in base_cols: rec[c] = df_used.iloc[i_doc][c]
            sample_rows.append(rec)

    pd.DataFrame(sample_rows).to_csv(out_dir / "samples_top_bottom.csv", index=False)
    print("[SAVE] samples_top_bottom.csv")

    # 9) 保存模型
    lda_path = out_dir / f"lda_K{K}.bin"
    lda.save(lda_path.as_posix())
    print(f"[SAVE] LDA model -> {lda_path}")

    # 10) 复盘提示
    print("\n[DONE] 主要产物：")
    for p in [
        "lda_dict.dict",
        "tokens_filtered.npy",
        "df_used.parquet",
        "train_metrics.json",
        "topics_top_words.csv",
        "topics_summary.csv",
        "doc_topics.parquet",
        "samples_top_bottom.csv",
        f"lda_K{K}.bin"
    ]:
        print(" -", out_dir / p)   
    print("\n[TRACEBACK READY] 任何时候可以用 doc_topics.parquet 里的 orig_row 回连回 IN_PARQUET；"
          "如需回到更早期原表，用你之前的映射 parquet 再 join 一次即可。")

if __name__ == "__main__":
    main()
