import datasets
from datasets import load_dataset, Dataset
from utils import to_pandas, convert_to_arrow
import pandas as pd
from gluonts.dataset.common import ListDataset
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Union
from datetime import datetime, timedelta
from gluonts.dataset.arrow import ArrowWriter
from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from huggingface_hub import HfApi

# Load the specific dataset
ds = load_dataset("autogluon/chronos_datasets", "exchange_rate", split="train")

def save_gluonts_dataset(dataset, save_path):
    """
    Guarda un ListDataset de GluonTS en un archivo CSV.

    Parameters:
    -----------
    dataset : ListDataset
        El dataset de GluonTS a guardar.
    save_path : str
        La ruta donde se guardará el archivo CSV.
    """
    data = []
    for entry in dataset:
        start = entry['start'].strftime('%Y-%m-%d %H:%M:%S')
        target = entry['target'].tolist()  # Convertir la serie a lista
        data.append({'start': start, 'target': target})

    # Crear un DataFrame y guardar como CSV
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

# Function to convert the general dataset to Arrow format
def convert_exchange_dataset_to_arrow(dataset, output_path):
  target_arrays = []

  # Iterate over the first 8 entries in the dataset
  for i in range(len(dataset)):
      targets = dataset[i]['target']  # Extract 'target' for each currency
      target_array = np.array(targets)  # Convert target list to NumPy array
      target_arrays.append(target_array)  # Append to list

  convert_to_arrow(
    path= output_path,
    time_series = target_arrays,
    compression = "lz4",
)

def crear_lista_fechas_laborales(start_date, target_length):
    # Crear la lista de fechas, filtrando solo días laborables
    lista = []
    current_date = start_date

    while len(lista) < target_length:
        # Verificar si la fecha es un día laboral según pandas
        if pd.Timestamp(current_date).weekday() < 5:  # 0-4 representa de lunes a viernes
            lista.append(current_date)
        current_date += timedelta(days=1)

    return lista

# Define a function to omit the last N data points
def omit_last_data_points(example):
    example['target'] = example['target'][:-num_points_to_omit]
    example['timestamp'] = example['timestamp'][:-num_points_to_omit]
    return example

def changingDatasetStructure(df):
  # Sort by 'timestamp' and 'id' to ensure consistent ordering
  df_sorted = df.sort_values(by=['timestamp', 'id'])

  # Step 1: Extract and flatten the 'target' values into a single list
  all_targets = []
  for targets in df_sorted['target']:
      # Check if 'targets' is iterable (like a list or numpy array)
      if isinstance(targets, (list, np.ndarray)):
          all_targets.extend(targets)  # Concatenate each list of target values
      else:
          all_targets.append(targets)  # Append if it's a single float value

  # Step 2: Create a new DataFrame with the flattened target values
  new_df = pd.DataFrame({'target': [all_targets]})

  # Get the length of the first row's 'target' list
  target_length = len(new_df.iloc[0]['target'])
  from datetime import datetime, timedelta

  # Ejemplo de uso
  fecha_inicial = datetime(1990, 1, 1, 0, 0)
  longitud_deseada = target_length
  lista = crear_lista_fechas_laborales(fecha_inicial, longitud_deseada)

  new_df['timestamp'] = [lista]  # Note that we wrap 'lista' in a list

  return new_df
import pyarrow as pa

def load_arrow_as_hf_dataset(arrow_file_path):
    # Read the Arrow file into a PyArrow table
    table = pa.ipc.RecordBatchFileReader(str(arrow_file_path)).read_all()

    # Convert the Arrow table into a Hugging Face Dataset
    hf_dataset = Dataset(pa.Table.from_batches([table.to_batches()[0]]))
    
    return hf_dataset

num_points_to_omit = 80

# Apply this function to each row in the dataset
new_ds = ds.map(omit_last_data_points, features=ds.features)

# Convert the Dataset to a pandas DataFrame
new_df = to_pandas(new_ds)

new_dataset = changingDatasetStructure(new_df)
new_ds2 = Dataset.from_pandas(new_dataset, split = ds.split, features = ds.features, info =ds.info)

from pathlib import Path

# Get the current file's directory
current_file_path = Path(__file__).resolve()
current_directory = current_file_path.parent

# Ensure the output directory exists
output_path = current_directory / "chronos-forecasting/scripts/evaluation/exchange_rate_new_dataset"
output_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist



# Convert to Arrow format
convert_exchange_dataset_to_arrow(new_ds2, str(current_directory / "datasets/exchange_rate_new_dataset.arrow"))

# Load the Arrow file into a Hugging Face Dataset
new_exchange_rate = load_arrow_as_hf_dataset(str(current_directory / "datasets/exchange_rate_new_dataset.arrow"))
# Save the dataset in Hugging Face format
new_exchange_rate.save_to_disk("./datasets/new_exchange_rate_dataset")

# Assuming `my_dataset` is the Dataset object you created
new_exchange_rate.save_to_disk("./datasets")

login(token="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")
from datasets import Dataset

from huggingface_hub import HfApi

# Create a new dataset repository (change the username and dataset name)
api = HfApi()
repo_url = api.create_repo(repo_id="juantollo/new_exchange_rate", repo_type="dataset", private=True, exist_ok=True)

# Upload the dataset files
api = HfApi()
api.upload_folder(
    folder_path="./datasets/new_exchange_rate_dataset",
    repo_id="juantollo/new_exchange_rate",
    repo_type="dataset"
)

