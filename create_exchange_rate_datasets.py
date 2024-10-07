import pdb
from datasets import load_dataset, Dataset, load_from_disk
from utils import to_pandas, convert_to_arrow
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Union
from datetime import datetime, timedelta
from gluonts.dataset.common import ListDataset
from huggingface_hub import login, HfApi
import pyarrow as pa
import os

# Load the specific dataset
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

def load_arrow_as_hf_dataset(arrow_file_path):
    # Read the Arrow file into a PyArrow table
    table = pa.ipc.RecordBatchFileReader(str(arrow_file_path)).read_all()
    # Convert the Arrow table into a Hugging Face Dataset
    hf_dataset = Dataset(pa.Table.from_batches([table.to_batches()[0]]))
    return hf_dataset

def filter_data_points(original_ds, num_points_to_omit):
    
    def omit_last_data_points(example):
        example['target'] = example['target'][:-num_points_to_omit]
        example['timestamp'] = example['timestamp'][:-num_points_to_omit]
        return example

    # Apply this function to each row in the dataset
    new_ds = ds.map(omit_last_data_points, features=ds.features)
    return new_ds

# Get the current file's directory
current_file_path = Path(__file__).resolve()
current_directory = current_file_path.parent

ds = load_dataset("autogluon/chronos_datasets", "exchange_rate", split="train")

################################################ TRAINING NEW DATASET ###################################################
directory_path = "datasets/training/"
os.makedirs(directory_path, exist_ok=True)

new_ds = filter_data_points(ds, 80)

# Convert the Dataset to a pandas DataFrame
new_df = to_pandas(new_ds)
new_dataset = changingDatasetStructure(new_df)
new_ds2 = Dataset.from_pandas(new_dataset, split = ds.split, features = ds.features, info =ds.info)

# Convert to Arrow format new dataset
convert_exchange_dataset_to_arrow(new_ds2, str(current_directory / "datasets/training/newExchangeRateTraining.arrow"))

################################################ TRAINING ORIGINAL DATASET ###################################################

# Convert to Arrow format old dataset
convert_exchange_dataset_to_arrow(ds, str(current_directory / "datasets/training/exchangeRateTraining.arrow"))

################################################ EVALUATION NEW DATASET ###################################################
### FIX BUG ###
#new_ds_evaluation = filter_data_points(ds, 80)

# Convert the Dataset to a pandas DataFrame
new_df_evaluation = to_pandas(ds)

new_dataset_evaluation = changingDatasetStructure(new_df_evaluation)

new_dataset_evaluation = Dataset.from_pandas(new_dataset_evaluation, split = ds.split, features = ds.features, info =ds.info)

print(type(new_dataset_evaluation))
# Save the dataset in Hugging Face format

new_dataset_evaluation.save_to_disk("./datasets/evaluation/newExchangeRate_evaluation")

login(token="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")

# Create a new dataset repository (change the username and dataset name)
api = HfApi()
repo_url = api.create_repo(repo_id="juantollo/newExchangeRate", repo_type="dataset", private=True, exist_ok=True)

new_dataset_evaluation.push_to_hub("juantollo/newExchangeRate")

################################################ TEST ################################################ 

# Convert the Dataset to a pandas DataFrame
df_evaluation = to_pandas(ds)

df_evaluation.to_csv("./datasets/evaluation/original_excgange_rate.csv")

new_dataset_evaluation = changingDatasetStructure(df_evaluation)

new_dataset_evaluation.to_csv("./datasets/evaluation/new_exchange_rate.csv")