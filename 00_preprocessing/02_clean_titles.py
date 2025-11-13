import os
import re
import spacy

nlp = spacy.load("en_core_web_sm")

input_folder = os.path.expanduser("~/Desktop/108家连续4年数据TXT（去掉页码噪音）")
cleaned_output_folder = os.path.join(input_folder, "cleaned_txts")
removed_titles_folder = os.path.join(input_folder, "removed_titles")
os.makedirs(cleaned_output_folder, exist_ok=True)
os.makedirs(removed_titles_folder, exist_ok=True)

allowed_lowercase = {"to", "of", "and", "for", "in", "on", "with", "at", "by", "the", "a", "an"}
months = {"january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"}

title_patterns = [
    r"^Risk(s)? Related to ",
    r"^Risks Associated With ",
    r"^General Risks$",
    r"^Legal Proceedings$",
    r"^Regulatory Risks$",
    r"^Risk Factor Summary$",
    r"^Summary of Risk Factors$",
    r"^Certain Risks? Relating to ",
]

def is_noun_phrase(text):
    doc = nlp(text.strip())
    return bool(doc) and doc[0].pos_ in {"NOUN", "PROPN"}

def contains_date_or_number(candidate):
    tokens = candidate.lower().split()
    return any(
        w.isdigit() or
        re.match(r"^\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?$", w) or
        w in months
        for w in tokens
    )

def detect_all_caps_titles(paragraph, removed_titles):
    pattern = r"(?<=[.;])\s+([A-Z][A-Z\s]{3,100})(?=\s+[A-Z])"
    match = re.search(pattern, paragraph)
    if match:
        candidate = match.group(1).strip()
        words = candidate.split()
        long_words = [w for w in words if len(w) >= 3 and w.isalpha()]
        if (
            3 <= len(words) <= 8 and
            len(long_words) >= 2 and
            not contains_date_or_number(candidate) and
            all(w.isupper() or w.lower() in allowed_lowercase for w in words)
        ):
            removed_titles.append(candidate)
            return paragraph.replace(candidate, "").strip()
    return paragraph

def detect_embedded_title(text, removed_titles):
    words = text.strip().split()
    max_title_len = min(20, len(words))

    for start in range(0, 6):
        for end in range(start + 3, min(start + 15, len(words))):
            candidate = " ".join(words[start:end])
            body_candidate = " ".join(words[end:])
            matches_pattern = any(re.match(p, candidate) for p in title_patterns)
            capital_count = sum(1 for w in words[start:end] if w[0].isupper() and w.lower() not in allowed_lowercase)
            looks_like_title = candidate.istitle() and len(candidate.split()) <= 8

            if ((capital_count >= 3 or matches_pattern or looks_like_title)
                and is_noun_phrase(candidate)
                and not contains_date_or_number(candidate)
                and len(body_candidate.split()) >= 5
            ):
                removed_titles.append(candidate)
                return " ".join(filter(None, [ " ".join(words[:start]).strip(), body_candidate.strip() ]))

    return text.strip()

def split_paragraph_by_inline_titles(paragraph):
    pattern = r"(?<=[.;])\s+(Risks? (Related|Relating) to [A-Z][^.;]*)"
    matches = list(re.finditer(pattern, paragraph))
    if not matches:
        return [paragraph]
    parts, last_index = [], 0
    for m in matches:
        split_start = m.start(1)
        parts.append(paragraph[last_index:split_start].strip())
        last_index = split_start
    parts.append(paragraph[last_index:].strip())
    return [p for p in parts if p]

def clean_file(file_path):
    removed_titles = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = re.split(r"\n\s*\n", content)
    cleaned_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        sub_paragraphs = split_paragraph_by_inline_titles(para)

        for sub_para in sub_paragraphs:
            words = sub_para.replace("\n", " ").split()
            capital_count = sum(1 for w in words[:12] if w[0].isupper() and w.lower() not in allowed_lowercase)

            is_title_line = (
                capital_count >= 3 and
                len(words) <= 12 and
                not sub_para.strip().endswith((".", "!", "?", ";", ":")) and
                is_noun_phrase(sub_para) and
                not contains_date_or_number(sub_para)
            )

            if is_title_line:
                removed_titles.append(sub_para)
                continue

            sub_para = detect_embedded_title(sub_para, removed_titles)
            sub_para = detect_all_caps_titles(sub_para, removed_titles)
            cleaned_paragraphs.append(sub_para)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    with open(os.path.join(cleaned_output_folder, f"{base_name}_cleaned.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(cleaned_paragraphs))
    with open(os.path.join(removed_titles_folder, f"{base_name}_removed_titles.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(removed_titles))

    print(f"✅ Processed: {base_name}")

# 批量处理
for filename in os.listdir(input_folder):
    if filename.endswith(".txt") and not filename.startswith(".") and "cleaned" not in filename:
        full_path = os.path.join(input_folder, filename)
        clean_file(full_path)

print("\n🎉 所有文件处理完毕")
print(f"📂 清洗后正文输出至: {cleaned_output_folder}")
print(f"📂 被识别标题输出至: {removed_titles_folder}")
