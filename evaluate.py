# Verificar el contenido del directorio para asegurarnos de que estamos accediendo a la ruta correcta
!ls /content/chronos-forecasting/scripts/evaluation

# Escribir el contenido YAML en un archivo en la carpeta correcta
yaml_content = """
- name: exchange_rate
  hf_repo: autogluon/chronos_datasets
  offset: -240
  prediction_length: 240
  num_rolls: 1
"""
with open("/content/chronos-forecasting/scripts/evaluation/configs/custom_zero_shot.yaml", "w") as file:
    file.write(yaml_content)

# Cambio de directorio para ejecutar el script desde la ruta correcta
import os
os.chdir('/content/chronos-forecasting/scripts/evaluation')

dir = f'"{output_model}/run-0/checkpoint-final"'
results = f'"{output_model}/new_model_exchange_rate_results.csv"'
# Execute the evaluation script using the custom configuration
!python evaluate.py configs/custom_zero_shot.yaml {results} \
    --chronos-model-id {dir} \
    --batch-size 32 \
    --device cuda:0 \
    --num-samples 20