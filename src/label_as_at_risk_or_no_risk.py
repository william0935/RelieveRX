import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_FILE = "../data/model_data/feature_matrix.csv"
OUTPUT_FILE = "model/feature_matrix_labeled.csv"
ICD_COLUMNS = [f"diagnoses_icd_icd_code_{i}" for i in range(100)]
PRESCRIPTION_COLUMNS = [f"prescriptions_dose_val_rx_{i}" for i in range(100)]

OVERDOSE_CODES = {"T40.0", "T40.1", "T40.2", "T40.3", "T40.4", "T40.6"}
ESCALATION_MULTIPLIER = 3.0
SUBJECT_COL = "subject_id"  # replace with your actual subject ID column

# load feature matrix with progress bar
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

def check_overdose(row):
    for col in ICD_COLUMNS:
        if col in row.index:
            code = str(row[col]).strip()
            if code in OVERDOSE_CODES:
                return 1
    return 0

def escalation_metrics(row, multiplier=ESCALATION_MULTIPLIER):
    doses = [row[col] for col in PRESCRIPTION_COLUMNS if col in row.index and pd.notna(row[col]) and row[col] != -1]

    if len(doses) < 2:
        return 0  # not at risk by escalation

    # net escalation or local escalation
    if doses[-1] > doses[0]:
        return 1
    if any(doses[i] >= multiplier * doses[i-1] for i in range(1, len(doses))):
        return 1

    return 0

print("computing at-risk labels...")

# compute at-risk labels
df['at_risk'] = df.apply(lambda row: max(check_overdose(row), escalation_metrics(row)), axis=1)

# select only subject id and at_risk for output
output_df = df[[SUBJECT_COL, 'at_risk']]

output_df.to_csv(OUTPUT_FILE, index=False)
print(f"Labeled dataset saved to {OUTPUT_FILE}")

risk_counts = output_df['at_risk'].value_counts()
print("\nlabeling summary:")
print(risk_counts)
