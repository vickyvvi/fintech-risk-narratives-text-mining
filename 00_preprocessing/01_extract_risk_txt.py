import os
import pandas as pd
import csv

def parse_txt_to_csv(txt_folder, output_csv_path):
    data = []

    for file in os.listdir(txt_folder):
        if file.endswith(".txt"):
            path = os.path.join(txt_folder, file)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                content = " ".join(line.strip() for line in lines if line.strip())

            parts = file.replace('.txt', '').split('_')
            if len(parts) >= 3:
                company = parts[0]
                filing_type = parts[1]  # Only keep '10-K' or '20-F'
                year = parts[2][:4]
            else:
                company, filing_type, year = "Unknown", "Unknown", "Unknown"

            data.append({
                "Company": company,
                "FilingType": filing_type,
                "Year": year,
                "RiskFactors": content
            })

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
    print(f"✅ 成功保存为：{output_csv_path}")

# 运行
if __name__ == "__main__":
    input_folder = '/Users/yuan/Desktop/108家连续4年数据TXT（去掉页码噪音）'
    output_csv = '/Users/yuan/Desktop/risk_factors_Crunchbase_LSEG.csv'
    parse_txt_to_csv(input_folder, output_csv)
