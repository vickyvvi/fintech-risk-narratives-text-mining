# 0) 安装依赖（CUDA 12.4 适配 RTX 5090/50 系列）
!pip -q install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip -q install -U transformers sentence-transformers tqdm pandas pyarrow numpy

# 1) 导入
import os
import gc
import math
import json
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from contextlib import nullcontext

# 2) 配置
INPUT_CSV = "/content/sentences_dedup_loose_keep_risky.csv"  # 改成你的路径

TEXT_COL = "sentence_raw"   # 语料列名
INIT_BATCH_SIZE = 256       # 起始 batch，若 OOM 会自动减半
MAX_LENGTH = 256            # FinBERT 分析时的最大长度（过长截断）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 情绪模型（FinBERT）
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"  # 输出: positive / neutral / negative

# 向量模型（Sentence-Transformers）
# 384 维（轻量，够用且快）：
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# 如需 768 维：
# EMB_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# 是否将向量展平为 emb_0...emb_n 列（体积很大，通常不建议）
EXPORT_FLAT_EMB = False

# 输出前缀
OUT_PREFIX = os.path.splitext(INPUT_CSV)[0] + "_with_sentiment"

# 3) 基础设置（5090 上更快）
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")     # 加速 matmul/attention
    AMP_DTYPE = torch.bfloat16                     
    AMP_CTX = torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
else:
    AMP_CTX = nullcontext()

# 4) 读取数据
df = pd.read_csv(INPUT_CSV)
if TEXT_COL not in df.columns:
    raise ValueError(f"找不到文本列 {TEXT_COL}")

print("输入规模:", df.shape)
texts = df[TEXT_COL].fillna("").astype(str).tolist()

# 5) 加载模型

# 5.1 FinBERT 情绪
finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
finbert_model.to(DEVICE)
finbert_model.eval()
# 注意：不同权重的 id->label 顺序可能不同，统一映射到 positive/neutral/negative
id2label = finbert_model.config.id2label
# 反转成 label->id，便于查列顺序
label2id = {v.lower(): k for k, v in id2label.items()}
# 目标顺序（统一）
target_order = ["positive", "neutral", "negative"]
missing = [lab for lab in target_order if lab not in label2id]
if missing:
    raise RuntimeError(f"FinBERT 标签不包含: {missing}（已获取 id2label={id2label}）")

# 5.2 句向量模型
emb_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)
emb_dim = emb_model.get_sentence_embedding_dimension()
print(f"Embedding model: {EMB_MODEL_NAME}, dim={emb_dim}")

# 6) FinBERT 情绪推理（带 OOM 自适应）
def finbert_predict(batch_texts):
    enc = finbert_tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE, non_blocking=True) for k, v in enc.items()}
    with torch.inference_mode(), AMP_CTX:
        logits = finbert_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)  # [B, C]
        pred_ids = probs.argmax(dim=-1).tolist()
        pred_labels = [id2label[i].lower() for i in pred_ids]
        probs_np = probs.detach().cpu().numpy()
    return pred_labels, probs_np

def run_finbert_all(text_list, init_bs=INIT_BATCH_SIZE):
    N = len(text_list)
    all_labels = []
    all_probs = []
    bs = init_bs
    i = 0
    pbar = tqdm(total=N, desc="FinBERT sentiment")
    while i < N:
        ok = False
        while not ok:
            try:
                batch = text_list[i:i+bs]
                labels, probs = finbert_predict(batch)
                all_labels.extend(labels)
                all_probs.append(probs)
                i += bs
                pbar.update(len(batch))
                ok = True
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    new_bs = max(8, bs // 2)
                    if new_bs == bs:
                        raise
                    print(f"[FinBERT] OOM：batch 从 {bs} -> {new_bs}")
                    bs = new_bs
                else:
                    raise
    pbar.close()
    all_probs = np.vstack(all_probs) if all_probs else np.zeros((0, len(id2label)))
    return all_labels, all_probs

finbert_labels, finbert_probs = run_finbert_all(texts, init_bs=INIT_BATCH_SIZE)

# 将概率列对齐为 positive/neutral/negative 顺序
# 先拿原序列各列在 target_order 中的位置
orig_order = [id2label[i].lower() for i in range(len(id2label))]
reorder_index = [orig_order.index(lab) for lab in target_order]  # 例如把 [neutral,pos,neg] -> [pos,neutral,neg]
finbert_probs_reordered = finbert_probs[:, reorder_index]

df["sentiment_label"] = finbert_labels
df["sentiment_score_positive"] = finbert_probs_reordered[:, 0]
df["sentiment_score_neutral"]  = finbert_probs_reordered[:, 1]
df["sentiment_score_negative"] = finbert_probs_reordered[:, 2]

# 7) 句向量（带 OOM 自适应）
def encode_embeddings(text_list, init_bs=INIT_BATCH_SIZE):
    # SentenceTransformer 内部也会分批；为更稳，这里再控制一层 batch（遇 OOM 降批）
    bs = init_bs
    N = len(text_list)
    out = np.zeros((N, emb_dim), dtype=np.float32)
    i = 0
    pbar = tqdm(total=N, desc="Embeddings")
    while i < N:
        ok = False
        while not ok:
            try:
                batch = text_list[i:i+bs]
                emb = emb_model.encode(
                    batch,
                    batch_size=bs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
                out[i:i+bs] = emb
                i += bs
                pbar.update(len(batch))
                ok = True
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    new_bs = max(8, bs // 2)
                    if new_bs == bs:
                        raise
                    print(f"[Embedding] OOM：batch 从 {bs} -> {new_bs}")
                    bs = new_bs
                else:
                    raise
    pbar.close()
    return out

embeddings = encode_embeddings(texts, init_bs=INIT_BATCH_SIZE)

# 8) 写出（Parquet 含 list；CSV 预览）
df["embedding"] = [emb.tolist() for emb in embeddings]

if EXPORT_FLAT_EMB:
    emb_cols = {f"emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
    emb_df = pd.DataFrame(emb_cols)
    df_flat = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)
else:
    df_flat = df.drop(columns=["embedding"]).copy()

# 8.1 含向量（list）+ 情绪（推荐 parquet）
df.to_parquet(OUT_PREFIX + ".parquet", index=False)

# 8.2 不含向量，仅情绪与原列（轻量预览）
df_flat.to_csv(OUT_PREFIX + ".csv", index=False)

# 8.3 可选：向量展平（大文件）
if EXPORT_FLAT_EMB:
    df_flat.to_csv(OUT_PREFIX + "_flat.csv", index=False)

print("已保存：")
print(OUT_PREFIX + ".parquet  （含情绪 + 向量list，更适合后续读写）")
print(OUT_PREFIX + ".csv      （仅情绪与原列，不含embedding列，便于预览）")
if EXPORT_FLAT_EMB:
    print(OUT_PREFIX + "_flat.csv （情绪 + 向量展平 emb_0..，体积较大，便于直接聚类）")

# 9) 清理
del embeddings
gc.collect()
if torch.cuda.is_available():

    torch.cuda.empty_cache()
