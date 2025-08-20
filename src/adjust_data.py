import pandas as pd
import os

# file paths
ANOMALY_FILE = "model/anomalies/vae_anomalies_subject_ids.csv"
DIAGNOSES_MAP_FILE = "../data/mimic_iv_data/mimic-iv-3.1/hosp/.csv_files/d_icd_diagnoses.csv"
PROCEDURES_MAP_FILE = "../data/mimic_iv_data/mimic-iv-3.1/hosp/.csv_files/d_icd_procedures.csv"
OUTPUT_FILE = "model/anomalies/vae_anomalies_with_text.csv"

# load files
print("Loading anomaly data...")
df = pd.read_csv(ANOMALY_FILE)

print("Loading ICD mapping files...")
diag_df = pd.read_csv(DIAGNOSES_MAP_FILE)
proc_df = pd.read_csv(PROCEDURES_MAP_FILE)

# normalize codes
def normalize_code(code):
    try:
        code = str(code).strip()
        if '.' in code:
            code = code.rstrip('0').rstrip('.')
        return code.lstrip('0') or '0'
    except:
        return str(code)

diag_df["icd_code"] = diag_df["icd_code"].apply(normalize_code)
proc_df["icd_code"] = proc_df["icd_code"].apply(normalize_code)

# build mappings from icd code to text
diag_map = {(row.icd_code, int(row.icd_version)): row.long_title for _, row in diag_df.iterrows()}
proc_map = {(row.icd_code, int(row.icd_version)): row.long_title for _, row in proc_df.iterrows()}

# some backup stuff
diag_fallback_map = {row.icd_code: row.long_title for _, row in diag_df.iterrows()}
proc_fallback_map = {row.icd_code: row.long_title for _, row in proc_df.iterrows()}

# get the column pairs
diagnosis_pairs = [
    (code_col, code_col.replace("icd_code", "icd_version"))
    for code_col in df.columns if code_col.startswith("diagnoses_icd_icd_code_")
]
procedure_pairs = [
    (code_col, code_col.replace("icd_code", "icd_version"))
    for code_col in df.columns if code_col.startswith("procedures_icd_icd_code_")
]

# apply mappings
def apply_icd_mapping(df, col_pairs, icd_map, fallback_map, label):
    for code_col, ver_col in col_pairs:
        def lookup(row):
            code = normalize_code(row.get(code_col))
            try:
                version = int(float(row.get(ver_col)))
            except:
                return "[UNKNOWN]"
            # try versioned lookup
            match = icd_map.get((code, version))
            if match:
                return match
            # try versionless fallback
            fallback = fallback_map.get(code)
            if fallback:
                return fallback
            return "[UNKNOWN]"
        df[code_col] = df.apply(lookup, axis=1)
        print(f"→ Replaced {code_col} with description from {label}")
    return df

print("Mapping diagnosis codes...")
df = apply_icd_mapping(df, diagnosis_pairs, diag_map, diag_fallback_map, "diagnoses")

print("Mapping procedure codes...")
df = apply_icd_mapping(df, procedure_pairs, proc_map, proc_fallback_map, "procedures")

# save
print(f"\nSaving output to: {OUTPUT_FILE}")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print("Done.")
