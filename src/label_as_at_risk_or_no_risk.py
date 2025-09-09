import pandas as pd
import numpy as np
from tqdm import tqdm
import json

## 1. globals
INPUT_FILE = "../data/model_data/feature_matrix.csv"
OUTPUT_FILE = "model/feature_matrix_labeled.csv"
SUBJECT_COL = "subject_id"

# columns
ICD_COLUMNS = [f"diagnoses_icd_icd_code_{i}" for i in range(100)]

# overdose ICD codes (poisoning by opioids)
OVERDOSE_CODES = {
    "T400", "T401", "T402", "T403", "T404",
    "T405", "T406", "T407", "T408", "T409"
}

## 2. load category mapping for ICD
with open("../data/model_data/category_mappings.json", "r") as f:
    category_mappings = json.load(f)

icd_map = category_mappings["diagnoses_icd_icd_code"]

## 3. load feature matrix
print("loading feature matrix with progress bar...")
total_lines = sum(1 for _ in open(INPUT_FILE)) - 1
chunksize = 10000

chunks = []
with tqdm(total=total_lines, desc="Reading CSV") as pbar:
    for chunk in pd.read_csv(INPUT_FILE, chunksize=chunksize):
        chunks.append(chunk)
        pbar.update(len(chunk))

df = pd.concat(chunks, ignore_index=True)
print("loaded feature matrix!")

## 4. compute overdose risk
def compute_overdose_flag(row):
    overdose_flag = 0
    for col in ICD_COLUMNS:
        if col in row and pd.notna(row[col]) and row[col] != -1:
            try:
                code_index = str(int(row[col]))
            except Exception:
                code_index = str(row[col]).strip()
            code = icd_map.get(code_index, "")
            if isinstance(code, str) and any(code.startswith(prefix) for prefix in OVERDOSE_CODES):
                overdose_flag = 1
                break
    return pd.Series({
        "at_risk": overdose_flag,      # now just overdose
        "overdose_flag": overdose_flag
    })

## 5. apply labeling
print("computing overdose labels...")
risk_df = df.apply(compute_overdose_flag, axis=1)
df[['at_risk', 'overdose_flag']] = risk_df

## 6. save dataset
output_df = df[[SUBJECT_COL, 'at_risk']]
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"labeled dataset saved to {OUTPUT_FILE}")

## 7. summary
total_at_risk = int(df['at_risk'].sum())
nothing = int((df['at_risk'] == 0).sum())

print("\nlabeling summary:")
print(f"total at risk (overdose): {total_at_risk}")
print(f"nothing: {nothing}")
