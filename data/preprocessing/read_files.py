import os
import pandas as pd

base_dir = 'data/mimic-iv-3.1/icu'
output_dir = os.path.join(base_dir, '__icu_outputs__')
n_lines = 100  # Number of lines to output

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for foldername in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, foldername)

    # Check if it's a directory and ends with '.csv'
    if os.path.isdir(folder_path) and foldername.endswith('.csv'):
        csv_file = os.path.join(folder_path, foldername)

        # Check if the CSV file exists
        if os.path.exists(csv_file):
            output_file = os.path.join(output_dir, foldername.replace('.csv', '_output.csv'))

            try:
                with open(csv_file, 'r') as infile, open(output_file, 'w') as outfile:
                    for i, line in enumerate(infile):
                        if i >= n_lines:
                            break
                        outfile.write(line)
                print(f"Successfully processed {csv_file} and created {output_file}")

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        else:
            print(f"CSV file not found in {folder_path}")