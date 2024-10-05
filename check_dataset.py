

import pyarrow as pa
from datasets import Dataset
import numpy as np

# Function to load Arrow dataset
def load_arrow_as_hf_dataset(arrow_file_path):
    table = pa.ipc.RecordBatchFileReader(str(arrow_file_path)).read_all()
    hf_dataset = Dataset(pa.Table.from_batches([table.to_batches()[0]]))
    return hf_dataset

# Path to the Arrow file
arrow_file_path = "./datasets/exchange_rate_new_dataset.arrow"

# Load the Arrow file into a Hugging Face Dataset
new_exchange_rate = load_arrow_as_hf_dataset(arrow_file_path)

# Extract the first 100 items in the target series
target_series = new_exchange_rate[0]['target'][:100]

# Print the first 100 items to check for separators
#print(target_series)

import json

import chardet

with open('/home/juantollo/foundational_models/datasets/exchange_rate_new_dataset.arrow', 'rb') as file:
    raw_data = file.read(10000)  # Read a chunk to detect encoding
    result = chardet.detect(raw_data)
    print(result)
