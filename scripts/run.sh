#!/bin/bash

# Check if the model parameter is passed
if [ -z "$1" ]; then
  echo "Error: Model size parameter is required."
  exit 1
fi

# Store the model parameter
model=$1

# Get the directory of the script
script_dir=$(dirname "$0")

# Navigate to the script's directory
cd "$script_dir"

# Change directory to ../chronos_forecasting/scripts
cd ../chronos-forecasting/scripts

# Path to the YAML configuration file
config_file="training/configs/google_t5-efficient-${model}.yaml"

# Check if the YAML file exists
if [ ! -f "$config_file" ]; then
  echo "Error: YAML configuration file not found: $config_file"
  exit 1
fi

echo "Running training script..."
python training/train.py --config "${config_file}" || { echo "Failed to run train.py"; exit 1; }

# Extract the output_dir from the YAML file using Python
output_dir=$(python3 -c "
import yaml
with open('${config_file}', 'r') as f:
    config = yaml.safe_load(f)
    print(config['output_dir'])
")

echo "Uploading new model..."
python "../../foundational_models/upload_fonder.py" "${output_dir}" || { echo "Failed to upload model folder"; exit 1; }
