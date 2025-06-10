import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from tqdm import tqdm

# === CONFIGURATION ===

INPUT_FOLDER = "mimic_iv_data/mimic-iv-3.1/hosp/.csv_files"
OUTPUT_FILE = "vae_data/vae_input_hosp.csv"
ROWS_PER_PATIENT_PER_FILE = 100

INCLUDED_COLUMNS_PER_FILE = {
    "diagnoses_icd": ["icd_code", "icd_version"],
    "procedures_icd": ["icd_code", "icd_version"],
    "prescriptions": ["drug", "dose_val_rx", "dose_unit_rx", "route"],
    "labevents": ["itemid", "valuenum", "valueuom", "flag", "priority"],
    "emar": ["medication", "route"],
    "microbiologyevents": ["spec_type_desc", "test_name", "org_name", "interpretation"],
    "omr": ["result_name", "result_value"],
    "pharmacy": ["medication", "dose_val_rx", "route"],
    "drgcodes": ["drg_type", "drg_code", "description", "drg_severity", "drg_mortality"],
    "services": ["curr_service"],
    "transfers": ["eventtype", "careunit"],
    "emar_detail": ["dose_due", "dose_given", "route", "product_description"],
    "poe": ["order_type", "order_status"],
    "poe_detail": ["field_value"],
    "hcpcsevents": ["hcpcs_cd", "short_description"],
    "admissions": ["admission_type", "insurance", "marital_status", "race", "hospital_expire_flag"],
    "patients": ["gender", "anchor_age", "anchor_year", "anchor_year_group"]
}

RELATIVE_TIME_COLUMNS = {
    "charttime", "storetime", "starttime", "stoptime", "chartdate", "edregtime", "edouttime",
    "ordertime", "eventtime", "intime", "outtime"
}

DURATION_COLUMNS = [
    ("starttime", "stoptime"),
    ("intime", "outtime"),
]

ADMISSIONS_FILE = "admissions.csv"

def parse_datetime_safe(x):
    try:
        return pd.to_datetime(x)
    except:
        return pd.NaT

def compute_relative_time(df, time_col, new_col_name):
    df[time_col] = df[time_col].apply(parse_datetime_safe)
    df[new_col_name] = (df[time_col] - df["baseline_time"]).dt.total_seconds() / 3600
    return df.drop(columns=[time_col])

def compute_duration(df, start_col, end_col, new_col_name):
    df[start_col] = df[start_col].apply(parse_datetime_safe)
    df[end_col] = df[end_col].apply(parse_datetime_safe)
    df[new_col_name] = (df[end_col] - df[start_col]).dt.total_seconds() / 3600
    return df.drop(columns=[start_col, end_col])

def flatten_patient_data(df, file_prefix):
    included_cols = INCLUDED_COLUMNS_PER_FILE.get(file_prefix, [])
    if not included_cols:
        print(f"Skipping {file_prefix} — no included columns listed.")
        return pd.DataFrame()

    all_subjects = []

    for subject_id, group in df.groupby("subject_id"):
        padded = group.iloc[:ROWS_PER_PATIENT_PER_FILE].copy()
        if len(padded) < ROWS_PER_PATIENT_PER_FILE:
            padding = pd.DataFrame(np.nan, index=range(ROWS_PER_PATIENT_PER_FILE - len(padded)), columns=group.columns)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                padded = pd.concat([padded, padding], ignore_index=True)

        padded["subject_id"] = subject_id
        all_subjects.append(padded)

    df = pd.concat(all_subjects)

    out = {}
    for subject_id, group in df.groupby("subject_id"):
        row = {}
        for col in included_cols:
            if col not in group.columns:
                continue
            values = group[col].tolist()
            for i, val in enumerate(values):
                row[f"{file_prefix}_{col}_{i}"] = val
        out[subject_id] = row

    df_out = pd.DataFrame.from_dict(out, orient="index")

    col_groups = {}
    for col in df_out.columns:
        if col.endswith(tuple(str(i) for i in range(ROWS_PER_PATIENT_PER_FILE))):
            *prefix_parts, index_str = col.rsplit("_", 1)
            base = "_".join(prefix_parts)
        else:
            base = col
            index_str = "0"
        col_groups.setdefault(base, []).append((col, int(index_str)))

    ordered_cols = []
    for base in sorted(col_groups):
        ordered_cols.extend([col for col, _ in sorted(col_groups[base], key=lambda x: x[1])])

    return df_out[ordered_cols]

# === LOAD ADMISSIONS FOR BASELINE TIME ===
print("Loading admissions data for baseline times...")
admissions_path = os.path.join(INPUT_FOLDER, ADMISSIONS_FILE)
admissions = pd.read_csv(admissions_path, parse_dates=["admittime"])
baseline_times = admissions[["subject_id", "admittime"]].dropna()
baseline_times = baseline_times.groupby("subject_id").first().rename(columns={"admittime": "baseline_time"})

merged_data = baseline_times.copy()

# === PROCESS EACH FILE ===
files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv") and f != ADMISSIONS_FILE])

for filename in tqdm(files, desc="Processing files"):  # <-- tqdm added here
    filepath = os.path.join(INPUT_FOLDER, filename)
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue

    if "subject_id" not in df.columns:
        continue

    df = df.merge(baseline_times, on="subject_id", how="inner")

    for time_col in RELATIVE_TIME_COLUMNS:
        if time_col in df.columns:
            df = compute_relative_time(df, time_col, f"{time_col}_since_admit_hours")

    for start_col, end_col in DURATION_COLUMNS:
        if start_col in df.columns and end_col in df.columns:
            df = compute_duration(df, start_col, end_col, f"{start_col}_to_{end_col}_duration_hours")

    file_prefix = os.path.splitext(filename)[0].lower()

    flattened = flatten_patient_data(df, file_prefix)
    merged_data = merged_data.merge(flattened, left_index=True, right_index=True, how="outer")

# === WRITE OUTPUT ===
merged_data = merged_data.drop(columns=["baseline_time"], errors="ignore")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
merged_data.reset_index().rename(columns={"index": "subject_id"}).to_csv(OUTPUT_FILE, index=False)
print(f"Flattened data saved to {OUTPUT_FILE}")
