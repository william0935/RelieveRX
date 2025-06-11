import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
import psutil

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
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df[new_col_name] = (df[time_col] - df["baseline_time"]).dt.total_seconds() / 3600
    df.drop(columns=[time_col], inplace=True)
    return df

def compute_duration(df, start_col, end_col, new_col_name):
    df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
    df[end_col] = pd.to_datetime(df[end_col], errors='coerce')
    df[new_col_name] = (df[end_col] - df[start_col]).dt.total_seconds() / 3600
    df.drop(columns=[start_col, end_col], inplace=True)
    return df

def flatten_patient_groups(df, file_prefix):
    included_cols = INCLUDED_COLUMNS_PER_FILE.get(file_prefix, [])
    if not included_cols:
        return []

    result = []
    for subject_id, group in df.groupby("subject_id"):
        padded = group.iloc[:ROWS_PER_PATIENT_PER_FILE].copy()
        if len(padded) < ROWS_PER_PATIENT_PER_FILE:
            pad_len = ROWS_PER_PATIENT_PER_FILE - len(padded)
            padding = pd.DataFrame(np.nan, index=range(pad_len), columns=group.columns)
            padded = pd.concat([padded, padding], ignore_index=True)

        row = {"subject_id": subject_id}
        for col in included_cols:
            if col not in padded.columns:
                continue
            vals = padded[col].tolist()
            for i, val in enumerate(vals):
                row[f"{file_prefix}_{col}_{i}"] = val
        result.append(row)
    return result

def process_file(filepath, file_prefix, baseline_times, output_handle):
    print(f"Processing {file_prefix}...")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Failed to load {file_prefix}: {e}")
        return

    if "subject_id" not in df.columns:
        print(f"Skipping {file_prefix}: no subject_id column.")
        return

    df = df.merge(baseline_times, on="subject_id", how="inner")

    for time_col in RELATIVE_TIME_COLUMNS:
        if time_col in df.columns:
            df = compute_relative_time(df, time_col, f"{time_col}_since_admit_hours")

    for start_col, end_col in DURATION_COLUMNS:
        if start_col in df.columns and end_col in df.columns:
            df = compute_duration(df, start_col, end_col, f"{start_col}_to_{end_col}_duration_hours")

    flattened = flatten_patient_groups(df, file_prefix)
    flat_df = pd.DataFrame(flattened)

    flat_df.to_csv(output_handle, mode='a', header=output_handle.tell() == 0, index=False)

def main():
    print("Loading baseline times...")
    admissions_path = os.path.join(INPUT_FOLDER, ADMISSIONS_FILE)
    admissions = pd.read_csv(admissions_path, parse_dates=["admittime"])
    baseline_times = admissions[["subject_id", "admittime"]].dropna()
    baseline_times = baseline_times.groupby("subject_id").first().rename(columns={"admittime": "baseline_time"})
    baseline_times.reset_index(inplace=True)

    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv") and f != ADMISSIONS_FILE])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f_out:
        for fname in tqdm(files, desc="All Files"):
            file_prefix = os.path.splitext(fname)[0].lower()
            filepath = os.path.join(INPUT_FOLDER, fname)
            process_file(filepath, file_prefix, baseline_times, f_out)
            print(f"[{file_prefix}] Memory: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

    print(f"Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
