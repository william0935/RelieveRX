import pandas as pd
import numpy as np
from tqdm import tqdm
import json

## 1. globals
INPUT_FILE = "../data/model_data/feature_matrix.csv"
OUTPUT_FILE = "model/feature_matrix_labeled.csv"
ICD_COLUMNS = [f"diagnoses_icd_icd_code_{i}" for i in range(100)]
PRESCRIPTION_COLUMNS = [f"prescriptions_dose_val_rx_{i}" for i in range(100)]
DRUG_COLUMNS = [f"prescriptions_drug_{i}" for i in range(100)]
SUBJECT_COL = "subject_id"

# absolute MME increase threshold for escalation
ABSOLUTE_MME_THRESHOLD = 50.0  # adjust as needed

# overdose ICD codes
OVERDOSE_CODES = {
    "T400", "T401", "T402", "T403", "T404",
    "T405", "T406", "T407", "T408", "T409"
}

# opioid MME conversion factors
OPIOID_MME_FACTORS = {
    "Codeine": 0.15,
    "Hydrocodone": 1,
    "Hydromorphone": 5,
    "Methadone (acute)": 3,
    "Methadone (chronic)": 4,
    "Morphine": 1,
    "Oxycodone": 1.5,
    "Oxymorphone": 10,
    "Tapentadol": 0.4,
    "Tramadol": 0.1
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

## 4. compute at risk flags
def compute_at_risk(row):
    # overdose ICD check
    overdose_flag = 0
    for col in ICD_COLUMNS:
        if col in row and pd.notna(row[col]) and row[col] != -1:
            code_index = str(int(row[col]))  # convert numeric index to string for mapping
            code = icd_map.get(code_index, "")
            if any(code.startswith(prefix) for prefix in OVERDOSE_CODES):
                overdose_flag = 1
                break

    # compute MME for escalation
    mme_list = []
    for drug_col, dose_col in zip(DRUG_COLUMNS, PRESCRIPTION_COLUMNS):
        if drug_col in row and dose_col in row and pd.notna(row[dose_col]) and row[dose_col] != -1:
            drug = str(row[drug_col]).strip()
            dose_val = row[dose_col]
            try:
                dose = float(dose_val)
                factor = OPIOID_MME_FACTORS.get(drug, 1.0)  # default factor=1 for unknown drugs
                mme = dose * factor
                mme_list.append(mme)
            except (ValueError, TypeError):
                continue  # skip invalid/missing

    # absolute escalation check
    escalation_flag = 0
    if len(mme_list) >= 2:
        mme_change = max(mme_list) - min(mme_list)
        if mme_change >= ABSOLUTE_MME_THRESHOLD:
            escalation_flag = 1

    # final at risk label
    at_risk = max(overdose_flag, escalation_flag)
    return pd.Series({
        "at_risk": at_risk,
        "overdose_flag": overdose_flag,
        "escalation_flag": escalation_flag
    })

## 5. compute at risk
print("computing at risk labels...")
risk_df = df.apply(compute_at_risk, axis=1)
df[['at_risk', 'overdose_flag', 'escalation_flag']] = risk_df

## 6. save dataset
output_df = df[[SUBJECT_COL, 'at_risk']]
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"labeled dataset saved to {OUTPUT_FILE}")

## 7. summary
total_at_risk = df['at_risk'].sum()
overdose_only = ((df['overdose_flag'] == 1) & (df['escalation_flag'] == 0)).sum()
escalation_only = ((df['overdose_flag'] == 0) & (df['escalation_flag'] == 1)).sum()
both_flags = ((df['overdose_flag'] == 1) & (df['escalation_flag'] == 1)).sum()
neither = ((df['at_risk'] == 0)).sum()

print("\nlabeling summary:")
print(f"total at risk: {total_at_risk}")
print(f"overdose only: {overdose_only}")
print(f"escalation only: {escalation_only}")
print(f"both overdose and escalation: {both_flags}")
print(f"neither: {neither}")
