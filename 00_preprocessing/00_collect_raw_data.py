%%writefile /content/txt_to_companyyear_csv.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate TXT -> CSV with 4 columns: Company, FilingType, Year, text
One row per (Company, FilingType, Year).

Filename patterns supported:
  A) <Company>_<FilingType>_<YYYYMMDD>.txt
  B) <Company>_<FilingType>_<Year>_<Date>.txt
Year rule:
  - Pattern B: take the 3rd segment's first 4 digits
  - Pattern A: take first 4 digits of date segment

Content rule:
  - Drop ONLY the line that starts with '##TITLE' (case-insensitive, leading spaces allowed)
  - Keep all other lines BEFORE/AFTER intact
  - Minimal cleaning: remove zero-width chars/BOM, tabs->space, collapse multiple spaces at ends
  - No other edits
Aggregation:
  - Concatenate kept lines per (Company, FilingType, Year), in file order then line order
  - Default joiner is newline; configurable via --join-with

Usage (Colab):
!python /content/txt_to_companyyear_csv.py \
  --in-dir /content \
  --out /content/text_merged_company_year.csv \
  --export-parquet
"""
import argparse, os, re, json
from pathlib import Path
import pandas as pd

TITLE_LINE_RE = re.compile(r"^\s*##title\b", re.IGNORECASE)
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")

def parse_fname(fname: str):
    """
    Parse:
      - <Company>_<FilingType>_<YYYYMMDD>.txt
      - <Company>_<FilingType>_<Year>_<Date>.txt
    Return: (Company, FilingType, Year) or (None, None, None)
    """
    stem = Path(fname).stem
    parts = stem.split("_")

    if len(parts) >= 4:
        company = parts[0]
        filing  = parts[1].upper()
        m = re.match(r"(\d{4})", parts[2])
        year = int(m.group(1)) if m else None
        return company, filing, year
    elif len(parts) == 3:
        company = parts[0]
        filing  = parts[1].upper()
        date    = parts[2]
        year = int(date[:4]) if re.fullmatch(r"\d{8}", date) else None
        return company, filing, year
    else:
        return None, None, None

def clean_line_minimal(s: str) -> str:
    s = ZERO_WIDTH_RE.sub("", s)
    s = s.replace("\t", " ")
    # 保持原文，避免过度清洗，只收尾压缩
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def gather_txts(in_dir: str, recursive: bool):
    p = Path(in_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")
    return sorted(p.rglob("*.txt")) if recursive else sorted(p.glob("*.txt"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=str, help="Directory containing .txt files.")
    ap.add_argument("--out",    required=True, type=str, help="Output CSV path.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--min-len", type=int, default=1, help="Skip lines shorter than this after trimming.")
    ap.add_argument("--join-with", type=str, default="\\n", help="Joiner between lines when aggregating (default newline). Use '\\n' or ' '.")
    ap.add_argument("--export-parquet", action="store_true", help="Also write Parquet.")
    args, _ = ap.parse_known_args()

    joiner = "\n" if args.join_with == "\\n" else args.join_with

    files = gather_txts(args.in_dir, args.recursive)
    if not files:
        raise FileNotFoundError(f"No .txt files found under: {args.in_dir}")

    # 收集每个 (Company, FilingType, Year) 的所有行
    buckets = {}  # key: (Company, FilingType, Year) -> list of lines
    audit = []    # file-level audit

    for fp in files:
        company, filing, year = parse_fname(fp.name)
        if company is None or filing is None or year is None:
            continue

        key = (company, filing, year)
        if key not in buckets:
            buckets[key] = []

        dropped = 0
        kept = 0
        with open(fp, "r", encoding="utf-8-sig", errors="ignore") as f:
            for raw in f:
                line = clean_line_minimal(raw.rstrip("\n\r"))
                if not line:
                    continue
                if TITLE_LINE_RE.match(line):
                    dropped += 1
                    continue
                if len(line) < args.min_len:
                    continue
                buckets[key].append(line)
                kept += 1

        audit.append({
            "filename": fp.name,
            "Company": company,
            "FilingType": filing,
            "Year": year,
            "dropped_title_lines": dropped,
            "kept_lines": kept
        })

    if not buckets:
        raise RuntimeError("No content aggregated. Check filenames/content.")

    # 按 key 聚合
    rows = []
    for (company, filing, year), lines in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        text = joiner.join(lines)
        rows.append({
            "Company": company,
            "FilingType": filing,
            "Year": year,
            "text": text
        })

    df = pd.DataFrame(rows, columns=["Company","FilingType","Year","text"])
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_parq = None
    if getattr(args, "export_parquet", False):
        out_parq = out_csv.with_suffix(".parquet")
        df.to_parquet(out_parq, index=False)

    audit_df = pd.DataFrame(audit)
    audit_path = out_csv.with_suffix(".audit.csv")
    audit_df.to_csv(audit_path, index=False)

    print(json.dumps({
        "files_processed": len(audit_df),
        "company_year_rows": len(df),
        "expected_note": "One row per (Company, FilingType, Year). If you expect 428, confirm count here."
    }, indent=2))
    print(f"CSV saved: {out_csv}")
    if out_parq:
        print(f"Parquet saved: {out_parq}")
    print(f"Audit saved: {audit_path}")
    print("Preview (first 2 rows):\n", df.head(2).to_string(index=False))

if __name__ == "__main__":
    main()



!python /content/txt_to_companyyear_csv.py \
  --in-dir /content \
  --out    /content/text_merged_company_year.csv \
  --export-parquet

import pandas as pd
df = pd.read_csv("/content/text_merged_company_year.csv")
len(df), df.head(12)

from google.colab import files
files.download("/content/text_merged_company_year.csv")




import pandas as pd

# 读取
df = pd.read_csv("/content/text_merged_company_year.csv")

print("总行数:", len(df))
print("公司数量:", df['Company'].nunique())
print("年份范围:", df['Year'].min(), "–", df['Year'].max())

# 每行文本长度（按字符数 & 单词数）
df["char_len"] = df["text"].astype(str).str.len()
df["word_len"] = df["text"].astype(str).str.split().map(len)

# 找最短/最长文本
shortest = df.nsmallest(5, "char_len")[["Company","FilingType","Year","char_len","word_len","text"]]
longest  = df.nlargest(5, "char_len")[["Company","FilingType","Year","char_len","word_len","text"]]

print("\n=== 最短的 5 行（按字符数）===")
print(shortest.to_string(index=False))

print("\n=== 最长的 5 行（按字符数）===")
print(longest.to_string(index=False))

# 统计整体分布
print("\n=== 长度分布（字符数）===")
print(df["char_len"].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]))

print("\n=== 长度分布（单词数）===")
print(df["word_len"].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]))

# 找出极短文本行（比如 <200 字符）
suspects = df[df["char_len"] < 200][["Company","FilingType","Year","char_len","word_len","text"]]
print("\n=== 可疑的极短文本（<200 字符）===")
print(suspects.head(10).to_string(index=False))
print("总数:", len(suspects))




import pandas as pd

# 路径按你当前文件
PATH_MAIN = "/content/text_merged_company_year.csv"
PATH_INFO = "/other_info.xlsx"
OUT_CSV   = "/content/text_merged_enriched.csv"
OUT_PQ    = "/content/text_merged_enriched.parquet"

# 1) 读入
df_main = pd.read_csv(PATH_MAIN)  # ['Company','FilingType','Year','text']
df_info = pd.read_excel(PATH_INFO)  # ['Company','Company Fullname','Sub-Sector','Stock Exchange','Headquarters location']

# 2) 只保留需要的列，避免同名冲突
cols_info_keep = ["Company","Company Fullname","Sub-Sector","Stock Exchange","Headquarters location"]
df_info = df_info[cols_info_keep].drop_duplicates("Company")  # 同公司多行时保留一条

# 3) 精确以 Company 合并（左连接，确保主表不丢行）
df_merged = df_main.merge(df_info, on="Company", how="left")

# 4) 保存
df_merged.to_csv(OUT_CSV, index=False)
df_merged.to_parquet(OUT_PQ, index=False)
print("✅ Saved:", OUT_CSV, "and", OUT_PQ)

# 5) 快速检查：哪些公司没匹配到补充信息
missing_mask = df_merged["Company Fullname"].isna() & df_merged["Sub-Sector"].isna() \
               & df_merged["Stock Exchange"].isna() & df_merged["Headquarters location"].isna()
missing_companies = (df_merged.loc[missing_mask, ["Company"]]
                     .drop_duplicates()
                     .sort_values("Company"))
print("\n未匹配到补充信息的公司数量：", len(missing_companies))
print(missing_companies.head(30).to_string(index=False))



import pandas as pd

# 读取合并后的 enriched 文件
df_enriched = pd.read_csv("/content/text_merged_enriched.csv")

# 打印总行数
print("总行数:", len(df_enriched))

# 打印前12行
print(df_enriched.head(12).to_string(index=False))

from google.colab import files
files.download("/content/text_merged_enriched.csv")
