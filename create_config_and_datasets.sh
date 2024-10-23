#!/bin/bash

# Step 4: Run the Python scripts to configure and create exchange rate datasets
echo "Running configuration script..."
python new_config_files.py || { echo "Failed to run new_config_files.py"; exit 1; }

echo "Creating exchange rate datasets..."
python create_exchange_rate_datasets.py || { echo "Failed to run create_excghange_rate_datasets.py"; exit 1; }
