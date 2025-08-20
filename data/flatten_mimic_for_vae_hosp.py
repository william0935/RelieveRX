import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import psutil
import warnings
import json

# configuration
USE_OUTPUT_FILES = False  # set to True to use output files, False to use original MIMIC-IV data
FILE_SUFFIX = "_output.csv" if USE_OUTPUT_FILES else ".csv"
ADMISSIONS_FILE = f"admissions{FILE_SUFFIX}"
INPUT_FOLDER = "preprocessing/__hosp_outputs__" if USE_OUTPUT_FILES else "mimic_iv_data/mimic-iv-3.1/hosp/.csv_files"
OUTPUT_FILE = "vae_data/vae_input_hosp_sample.csv" if USE_OUTPUT_FILES else "vae_data/vae_input_hosp.csv"
OUTPUT_CLIPPED_FOLDER = "vae_data/__vae_outputs__"
CATEGORY_MAPPING_FILE = "vae_data/category_mappings.json"
CLIPPED_LINES = 100
ROWS_PER_PATIENT_PER_FILE = 100
CHUNK_SIZE = 50000

INCLUDED_COLUMNS_PER_FILE = {
    "diagnoses_icd": ["icd_code", "icd_version"],
    "procedures_icd": ["icd_code", "icd_version"],
    "prescriptions": ["drug_type", "drug", "prod_strength", "dose_val_rx", "dose_unit_rx", "route", "starttime", "stoptime"],
    "labevents": ["itemid", "valuenum", "valueuom", "flag", "priority"],
    "emar": ["medication", "route"],
    "microbiologyevents": ["spec_type_desc", "test_name", "org_name", "interpretation"],
    "omr": ["result_name", "result_value"],
    "pharmacy": ["medication", "route"],
    "drgcodes": ["drg_type", "drg_code", "drg_severity", "drg_mortality"],
    "services": ["curr_service"],
    "transfers": ["eventtype", "careunit"],
    "emar_detail": ["dose_due", "dose_given", "route", "product_code"],
    "poe": ["order_type", "order_status"],
    "poe_detail": ["field_value"],
    "hcpcsevents": ["hcpcs_cd"],
    "admissions": ["admission_type", "insurance", "marital_status", "race", "hospital_expire_flag"],
    "patients": ["gender", "anchor_age", "anchor_year", "anchor_year_group"]
}

SPECIAL_MISSING_VALUE = -1
category_mappings = {}  # global dictionary to store category mappings

def print_memory_usage(prefix=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"{prefix}Memory usage: {mem:.2f} GB")

def flatten_patient_groups(df, file_prefix):
    included_cols = INCLUDED_COLUMNS_PER_FILE.get(file_prefix, [])
    if not included_cols:
        return pd.DataFrame()

    patient_rows = []
    for subject_id, group in tqdm(df.groupby("subject_id"), desc=f"Flattening {file_prefix}", leave=True):
        padded = group.iloc[:ROWS_PER_PATIENT_PER_FILE].copy()
        pad_len = ROWS_PER_PATIENT_PER_FILE - len(padded)
        if pad_len > 0:
            pad = {col: [np.nan] * pad_len for col in group.columns}
            pad_df = pd.DataFrame(pad)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                padded = pd.concat([padded, pad_df], ignore_index=True)

        row = {"subject_id": int(subject_id)}
        for col in included_cols:
            if col not in padded.columns:
                continue
            vals = padded[col].tolist()
            if pd.api.types.is_numeric_dtype(padded[col]):
                pass # keep NaNs, fill at the end
            else:
                cat_series = pd.Series(vals).astype("category")
                codes = cat_series.cat.codes.replace(-1, SPECIAL_MISSING_VALUE)
                vals = codes.tolist()
                col_key = f"{file_prefix}_{col}"
                category_mappings[col_key] = dict(enumerate(cat_series.cat.categories))
            for i, val in enumerate(vals):
                row[f"{file_prefix}_{col}_{i}"] = val
        patient_rows.append(row)

    return pd.DataFrame(patient_rows)

def process_file(filepath, file_prefix, baseline_times, accumulated):
    try:
        if "subject_id" not in pd.read_csv(filepath, nrows=1).columns:
            print(f"Skipping {file_prefix}: no subject_id")
            return
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
        chunk = chunk.merge(baseline_times, on="subject_id", how="left")
        chunk = chunk[chunk["baseline_time"].notna()]
        flat_chunk = flatten_patient_groups(chunk, file_prefix)
        if not flat_chunk.empty:
            accumulated.append(flat_chunk)
        print(f"  Processed {len(chunk)} rows in {file_prefix}")
        print_memory_usage(f"    After chunk in {file_prefix}: ")

def write_clipped_version(input_path, output_folder, max_lines=100):
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, filename.replace(".csv", "_output.csv"))
    with open(input_path, "r", buffering=1 << 16) as infile, open(output_path, "w", buffering=1 << 16) as outfile:
        for i, line in enumerate(infile):
            if i >= max_lines:
                break
            outfile.write(line)
    print(f"Clipped version written to: {output_path}")

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    admissions_path = os.path.join(INPUT_FOLDER, ADMISSIONS_FILE)
    admissions_df = pd.read_csv(admissions_path, usecols=["subject_id", "admittime"])
    admissions_df["baseline_time"] = pd.to_datetime(admissions_df["admittime"], errors="coerce")
    baseline_times = admissions_df[["subject_id", "baseline_time"]].copy()
    baseline_times["subject_id"] = pd.to_numeric(baseline_times["subject_id"], errors="coerce").astype("Int64")

    suffix = FILE_SUFFIX
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(suffix) and f != ADMISSIONS_FILE]
    all_flattened_chunks = []

    for fname in tqdm(csv_files, desc="Processing files"):
        file_prefix = fname.replace(FILE_SUFFIX, "")
        filepath = os.path.join(INPUT_FOLDER, fname)
        process_file(filepath, file_prefix, baseline_times, all_flattened_chunks)
        print_memory_usage(f"After {fname}: ")

    if all_flattened_chunks:
        final_df = pd.concat(all_flattened_chunks).groupby("subject_id").first().reset_index()
        final_df.fillna(SPECIAL_MISSING_VALUE, inplace=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Final flattened dataset written to: {OUTPUT_FILE}")

    with open(CATEGORY_MAPPING_FILE, "w") as f:
        json.dump(category_mappings, f)
        print(f"Category mappings saved to: {CATEGORY_MAPPING_FILE}")

    write_clipped_version(OUTPUT_FILE, OUTPUT_CLIPPED_FOLDER, max_lines=CLIPPED_LINES)

if __name__ == "__main__":
    main()
