import os

# === CONFIGURATION ===
base_dir = 'vae_data'
output_dir = os.path.join(base_dir, '__vae_outputs__')
n_lines = 100  # Number of lines to output per file

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each file in base_dir
for filename in os.listdir(base_dir):
    if not filename.endswith('.csv'):
        continue

    input_path = os.path.join(base_dir, filename)
    output_path = os.path.join(output_dir, filename.replace('.csv', '_output.csv'))

    if not os.path.isfile(input_path):
        continue

    try:
        with open(input_path, 'r', buffering=1 << 16) as infile, open(output_path, 'w', buffering=1 << 16) as outfile:
            for i, line in enumerate(infile):
                if i >= n_lines:
                    break
                outfile.write(line)
        print(f"Processed: {filename}")
    except Exception as e:
        print(f"Failed: {filename} -> {e}")
