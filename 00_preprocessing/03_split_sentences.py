#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_semicolons.py
分号二次切分（仅对足够长的句子），并保留回溯信息。
- 自动识别输入文件分隔符：逗号/分号/tab（可 --sep 覆盖）
- 尝试多种编码读取，必要时进行轻度清洗
- 输出：.csv（必有）与 .parquet（若环境可用）

新增：对子句末尾补分号；若已以 . ! ? ; 结尾则不补。
"""

import re
import argparse
import pandas as pd
from io import StringIO
from typing import Optional, Tuple, List

# -------------------- 基础文本工具 --------------------

def simple_tokenize(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\$€£]?\d+(?:[\.,]\d+)*%?", s)

def split_semicolons_with_spans(text: str) -> Tuple[List[str], List[Tuple[int,int]]]:
    parts, spans = [], []
    start = 0
    for m in re.finditer(r";", text):
        end = m.start()
        parts.append(text[start:end].strip())
        spans.append((start, end))
        start = m.end()
    parts.append(text[start:].strip())
    spans.append((start, len(text)))
    return parts, spans

def ensure_terminal_punct_semicolon(s: str) -> str:
    """若结尾没有 . ! ? ; 则补一个分号；已有这些标点则不补。"""
    s = s.rstrip()
    if not s:
        return s
    if s.endswith((".", "!", "?", ";")):
        return s
    # 若以引号/括号等结尾，也直接在后面补分号，不修改原字符
    return s + ";"

# -------------------- 自动分隔符识别与鲁棒读取 --------------------

COMMON_DELIMS = [",", ";", "\t"]

def detect_delimiter_from_text(sample: str) -> Optional[str]:
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    lines = lines[:50]
    if not lines:
        return None
    counts = {d: 0 for d in COMMON_DELIMS}
    for ln in lines:
        for d in COMMON_DELIMS:
            counts[d] += ln.count(d)
    if max(counts.values()) == 0:
        return None
    max_val = max(counts.values())
    candidates = [d for d, v in counts.items() if v == max_val]
    for pref in ["\t", ",", ";"]:
        if pref in candidates:
            return pref
    return candidates[0] if candidates else None

def try_read_csv(path: str, sep: Optional[str], encoding: Optional[str]):
    return pd.read_csv(path, sep=sep, encoding=encoding, engine="python", on_bad_lines="skip")

def robust_read_table(path: str, preferred_sep: Optional[str]=None, preferred_encoding: Optional[str]=None) -> Tuple[pd.DataFrame, str]:
    try:
        with open(path, "rb") as f:
            raw_head = f.read(200_000)
    except Exception as e:
        raise RuntimeError(f"Cannot open file: {path}: {e}")

    encodings = [preferred_encoding] if preferred_encoding else []
    encodings += ["utf-8", "utf-8-sig", "ascii", "cp1252", "latin1", "ISO-8859-1"]

    sample_text = None
    for enc in encodings:
        if enc is None: 
            continue
        try:
            sample_text = raw_head.decode(enc, errors="strict"); sample_enc = enc; break
        except Exception:
            continue
    if sample_text is None:
        sample_text = raw_head.decode("utf-8", errors="replace"); sample_enc = "utf-8"

    delim = preferred_sep or detect_delimiter_from_text(sample_text)
    try_seps = [delim] if delim else [None]
    for d in COMMON_DELIMS:
        if d not in try_seps:
            try_seps.append(d)

    for enc in [sample_enc] + [e for e in encodings if e and e != sample_enc]:
        for sep in try_seps:
            try:
                df = try_read_csv(path, sep=sep, encoding=enc)
                return df, (sep or "(infer)")
            except Exception:
                continue

    # fallback: clean then retry
    with open(path, "rb") as f:
        raw = f.read()
    txt = raw.decode("utf-8", errors="replace")
    txt = txt.replace("\x00", "")
    txt = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", txt)
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")

    delim2 = preferred_sep or detect_delimiter_from_text(txt)
    try_seps2 = [delim2] if delim2 else [None]
    for d in COMMON_DELIMS:
        if d not in try_seps2:
            try_seps2.append(d)

    for sep in try_seps2:
        try:
            df = pd.read_csv(StringIO(txt), sep=sep, engine="python", on_bad_lines="skip")
            return df, (sep or "(infer)")
        except Exception:
            continue

    raise RuntimeError("Failed to read table after robust attempts. Consider specifying --sep and --encoding explicitly.")

# -------------------- 分号二次切分核心逻辑 --------------------

def split_one_row(row, text_col: str,
                  min_tokens_piece: int = 6,
                  trigger_tokens: int = 40,
                  trigger_chars: int = 200):
    text = row.get(text_col, "")
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    tokens = simple_tokenize(text)
    char_len = len(text)

    if len(tokens) < trigger_tokens and char_len < trigger_chars:
        return [{
            "parent_sent_id": row["sent_id"],
            "sub_id": 1,
            "sub_sent_id": f'{row["sent_id"]}-01',
            "sentence_sub": ensure_terminal_punct_semicolon(text),
            "span_start": 0,
            "span_end": char_len,
            "parent_token_len": len(tokens),
            "parent_char_len": char_len,
        }]

    parts, spans = split_semicolons_with_spans(text)
    items = [{"txt": p, "span": (s0, s1), "tok": len(simple_tokenize(p))}
             for (p, (s0, s1)) in zip(parts, spans)]

    merged = []
    i = 0
    while i < len(items):
        cur = items[i]
        if cur["tok"] >= min_tokens_piece or len(items) == 1:
            merged.append(cur); i += 1
        else:
            if merged:
                prev = merged[-1]
                prev["txt"] = (prev["txt"] + " " + cur["txt"]).strip()
                prev["span"] = (prev["span"][0], cur["span"][1])
                prev["tok"] = len(simple_tokenize(prev["txt"]))
                i += 1
            elif i + 1 < len(items):
                nxt = items[i + 1]
                nxt["txt"] = (cur["txt"] + " " + nxt["txt"]).strip()
                nxt["span"] = (cur["span"][0], nxt["span"][1])
                nxt["tok"] = len(simple_tokenize(nxt["txt"]))
                i += 2
            else:
                merged.append(cur); i += 1

    # ★ 在输出前补分号（若未以 . ! ? ; 结尾）
    for it in merged:
        it["txt"] = ensure_terminal_punct_semicolon(it["txt"])

    out = []
    for j, it in enumerate(merged, 1):
        out.append({
            "parent_sent_id": row["sent_id"],
            "sub_id": j,
            "sub_sent_id": f'{row["sent_id"]}-{j:02d}',
            "sentence_sub": it["txt"],
            "span_start": it["span"][0],
            "span_end": it["span"][1],
            "parent_token_len": len(tokens),
            "parent_char_len": char_len,
        })
    return out

def split_dataframe(df: pd.DataFrame, text_col="sentence_raw",
                    min_tokens_piece=6, trigger_tokens=40, trigger_chars=200) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        pieces = split_one_row(
            r, text_col=text_col,
            min_tokens_piece=min_tokens_piece,
            trigger_tokens=trigger_tokens,
            trigger_chars=trigger_chars
        )
        base = r.to_dict()
        for d in pieces:
            x = dict(base); x.update(d); rows.append(x)

    cols_keep = list(df.columns) + [
        "parent_sent_id","sub_id","sub_sent_id","sentence_sub",
        "span_start","span_end","parent_token_len","parent_char_len"
    ]
    return pd.DataFrame(rows)[cols_keep]

# -------------------- 主程序 --------------------

def main():
    ap = argparse.ArgumentParser(description="Semicolon second-stage splitter with traceability (auto delimiter).")
    ap.add_argument("--input", required=True, help="输入文件路径（CSV/TSV）")
    ap.add_argument("--output", required=True, help="输出文件前缀（无需扩展名）")
    ap.add_argument("--text-col", default="sentence_raw", help="文本列名（默认：sentence_raw）")
    ap.add_argument("--id-col", default="sent_id", help="ID 列名（默认：sent_id；缺失时按行号自动生成）")
    ap.add_argument("--min-tokens-piece", type=int, default=6, help="子句最少 token 数（默认 6）")
    ap.add_argument("--trigger-tokens", type=int, default=40, help="触发切分的最小 token 数（默认 40）")
    ap.add_argument("--trigger-chars", type=int, default=200, help="触发切分的最小字符数（默认 200）")
    ap.add_argument("--sep", default=None, help="手动指定分隔符（可选：',' ';' '\\t'），留空则自动识别")
    ap.add_argument("--encoding", default=None, help="首选编码（留空自动尝试）")
    args = ap.parse_args()

    df, used_sep = robust_read_table(args.input, preferred_sep=args.sep, preferred_encoding=args.encoding)
    print(f"[info] Using delimiter: {repr(used_sep)}")

    if args.text_col not in df.columns:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not obj_cols:
            raise ValueError(f"Text column '{args.text_col}' not found and no object-like column available.")
        print(f"[warn] '{args.text_col}' not found. Falling back to first object column: {obj_cols[0]}")
        args.text_col = obj_cols[0]

    if args.id_col not in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": args.id_col})
        df[args.id_col] = df[args.id_col].astype(int) + 1

    renamed_id = False
    if args.id_col != "sent_id":
        df = df.rename(columns={args.id_col: "sent_id"})
        renamed_id = True

    df[args.text_col] = df[args.text_col].fillna("").astype(str)

    out_df = split_dataframe(
        df,
        text_col=args.text_col,
        min_tokens_piece=args.min_tokens_piece,
        trigger_tokens=args.trigger_tokens,
        trigger_chars=args.trigger_chars
    )

    if renamed_id:
        out_df = out_df.rename(columns={"sent_id": args.id_col, "parent_sent_id": args.id_col})

    base = args.output.rsplit(".", 1)[0]
    csv_path = base + ".csv"
    out_df.to_csv(csv_path, index=False)
    print(f"[ok] CSV written: {csv_path}")

    wrote_parquet = False
    for engine in ("pyarrow", "fastparquet"):
        try:
            out_df.to_parquet(base + ".parquet", index=False, engine=engine)
            print(f"[ok] Parquet written: {base + '.parquet'} (engine={engine})")
            wrote_parquet = True
            break
        except Exception:
            continue
    if not wrote_parquet:
        print("[info] pyarrow/fastparquet not available; skipped parquet.")

    split_counts = out_df.groupby(out_df.columns[out_df.columns.get_loc("parent_sent_id")])["sub_id"].max()
    total_parents = df.shape[0]
    total_rows_after = out_df.shape[0]
    num_parents_split = int((split_counts > 1).sum())
    avg_pieces_when_split = float(split_counts[split_counts > 1].mean()) if num_parents_split > 0 else 1.0
    share = round(num_parents_split / total_parents, 4) if total_parents else 0.0

    print("------ SUMMARY ------")
    print(f"input_rows (parents): {total_parents}")
    print(f"output_rows (subs):   {total_rows_after}")
    print(f"num_parents_split:    {num_parents_split}")
    print(f"share_parents_split:  {share}")
    print(f"avg_pieces_when_split:{round(avg_pieces_when_split, 3)}")

if __name__ == "__main__":
    main()
