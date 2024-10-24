import os
import pandas as pd

# Define the directory containing the CSV files
results_dir = 'evaluation/results'
output_file = 'evaluation/results/consolidated_results.csv'

# Sanity check: print the files in the directory
print(f"Checking files in the directory: {results_dir}")
files = os.listdir(results_dir)
print(f"Files found: {files}")

# Initialize an empty list to store the data
data = []

# Loop through all the CSV files in the results directory
for filename in files:
    if filename.endswith('.csv') and filename != 'consolidated_results.csv':  # Avoid consolidating the output file itself
        file_path = os.path.join(results_dir, filename)
        
        # Sanity check: print the file being read
        print(f"Reading file: {file_path}")
        
        # Open the CSV file and extract the relevant data
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('dataset'):
                    continue  # Skip the header line
                # Sanity check: print the line being processed
                print(f"Processing line: {line.strip()}")
                data.append(line.strip().split(','))

# If no data was collected, print a warning message
if not data:
    print("No data found in the CSV files. Please check the file contents.")

# Convert the data into a DataFrame if data exists
if data:
    df = pd.DataFrame(data, columns=['dataset', 'model', 'MASE', 'MAE[0.5]', 'MSE[mean]', 'WQL'])
    
    # Save the consolidated data to a CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Consolidated results saved to {output_file}.")
else:
    print("No data to save to the consolidated file.")
