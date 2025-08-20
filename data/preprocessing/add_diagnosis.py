import pandas as pd
import os

def update_frequency_file_with_text(frequency_file_path, description_file_path, code_column_freq, code_column_desc, text_column_desc, new_text_column_name):
    try:
        # read frequency file
        frequency_df = pd.read_csv(frequency_file_path)

        # read description file
        description_df = pd.read_csv(description_file_path)

        # ensure code columns are string type
        description_df[code_column_desc] = description_df[code_column_desc].astype(str)
        frequency_df[code_column_freq] = frequency_df[code_column_freq].astype(str)

        # create dictionary for code to text description lookup
        description_dict = pd.Series(description_df[text_column_desc].values, index=description_df[code_column_desc]).to_dict()

        # add new text column by mapping the code
        frequency_df[new_text_column_name] = frequency_df[code_column_freq].map(description_dict)

        # overwrite original frequency file
        frequency_df.to_csv(frequency_file_path, index=False)

        print(f"Successfully updated {frequency_file_path} with {new_text_column_name}.")

    except FileNotFoundError:
        print(f"Error: One or more input files not found. Please check paths:\n- {frequency_file_path}\n- {description_file_path}")
    except KeyError as e:
        print(f"Error: A required column was not found. It's likely '{e.args[0]}' is missing in one of the files or was misspelled in the arguments.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# files and directories for icd
base_dir_hosp = 'data/mimic-iv-3.1/hosp'
sorted_dir = os.path.join('.', 'sorted')
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

# drg codes
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