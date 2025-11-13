import os
import re

# 设置路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
input_folder = os.path.join(desktop_path, "108家连续4年数据TXT")
output_folder = os.path.join(desktop_path, "108家连续4年数据TXT_cleaned")

# 创建输出文件夹（如不存在）
os.makedirs(output_folder, exist_ok=True)

# 正则：匹配“数字 + Table of Contents”，允许换行、空格
toc_pattern = re.compile(
    r'(?:\d{1,3}[ \t]*(?:\n{0,2})[ \t]*)table of contents',
    re.IGNORECASE
)

# 遍历 .txt 文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".txt"):
        input_path = os.path.join(input_folder, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 检查是否匹配“数字 + TOC”
        if re.search(toc_pattern, content):
            # 删除所有匹配项
            new_content = re.sub(toc_pattern, "", content)

            # 保存到新文件夹，保留原文件名
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"✅ Cleaned and saved: {filename}")
        else:
            # 不保存没有匹配的文件
            print(f"⏭ Skipped (no TOC): {filename}")