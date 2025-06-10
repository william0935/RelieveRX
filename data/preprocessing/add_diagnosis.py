import pandas as pd
import os

def update_frequency_file_with_text(frequency_file_path, description_file_path, code_column_freq, code_column_desc, text_column_desc, new_text_column_name):
    """
    Reads a frequency file, looks up text descriptions from a description file
    based on a code, adds it as a new column, and overwrites the original frequency file.

    Args:
        frequency_file_path (str): Path to the frequency CSV file.
        description_file_path (str): Path to the CSV file containing code descriptions.
        code_column_freq (str): Name of the code column in the frequency file.
        code_column_desc (str): Name of the code column in the description file.
        text_column_desc (str): Name of the column containing the text description in the description file.
        new_text_column_name (str): Name for the new column to be added to the frequency file.
    """
    try:
        # Read the frequency file into a pandas DataFrame
        frequency_df = pd.read_csv(frequency_file_path)

        # Read the description file into a pandas DataFrame
        description_df = pd.read_csv(description_file_path)

        # Ensure code columns are string type for reliable mapping
        description_df[code_column_desc] = description_df[code_column_desc].astype(str)
        frequency_df[code_column_freq] = frequency_df[code_column_freq].astype(str)

        # Create a dictionary for code to text description lookup
        description_dict = pd.Series(description_df[text_column_desc].values, index=description_df[code_column_desc]).to_dict()

        # Add the new text column by mapping the code
        frequency_df[new_text_column_name] = frequency_df[code_column_freq].map(description_dict)

        # Overwrite the original frequency file with the updated DataFrame
        frequency_df.to_csv(frequency_file_path, index=False)

        print(f"Successfully updated {frequency_file_path} with {new_text_column_name}.")

    except FileNotFoundError:
        print(f"Error: One or more input files not found. Please check paths:\n- {frequency_file_path}\n- {description_file_path}")
    except KeyError as e:
        print(f"Error: A required column was not found. It's likely '{e.args[0]}' is missing in one of the files or was misspelled in the arguments.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Configuration ---
base_dir_hosp = 'data/mimic-iv-3.1/hosp'
sorted_dir = os.path.join('.', 'sorted')

# --- Processing for ICD codes ---
# ICD file paths
icd_frequency_file = os.path.join(sorted_dir, 'diagnoses_icd_frequency.csv')
icd_diagnosis_file = os.path.join(base_dir_hosp, 'd_icd_diagnoses.csv', 'd_icd_diagnoses.csv')

print("Processing ICD codes...")
update_frequency_file_with_text(
    frequency_file_path=icd_frequency_file,
    description_file_path=icd_diagnosis_file,
    code_column_freq='icd_code',
    code_column_desc='icd_code',
    text_column_desc='long_title',
    new_text_column_name='diagnoses'
)
print("-" * 30)

# --- Processing for DRG codes ---
# DRG file paths
drg_frequency_file = os.path.join(sorted_dir, 'drgcodes_frequency.csv')
drg_description_file = os.path.join(base_dir_hosp, 'drgcodes.csv', 'drgcodes.csv')

print("Processing DRG codes...")
update_frequency_file_with_text(
    frequency_file_path=drg_frequency_file,
    description_file_path=drg_description_file,
    code_column_freq='drg_code',
    code_column_desc='drg_code',
    text_column_desc='description',
    new_text_column_name='description'
)