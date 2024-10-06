################################################################# EVALUATION ###############################################################

new_exchange_rate_zero_shot = """
- name: exchange_rate
  hf_repo: juantollo/new_exchange_rate"
  offset: -240
  prediction_length: 240
  num_rolls: 1
"""

exchange_zero_shot = """
- name: exchange_rate
  hf_repo: autogluon/chronos_datasets
  offset: -30
  prediction_length: 30
  num_rolls: 1
"""

exchange_zero_shot_test = """
- name: exchange_rate
  hf_repo: autogluon/chronos_datasets
  offset: -10
  prediction_length: 10
  num_rolls: 1
"""

yalms = {"new_exchange_rate_zero_shot": new_exchange_rate_zero_shot,
         "exchange_zero_shot": exchange_zero_shot,
         "exchange_zero_shot_test": exchange_zero_shot_test
         } 

for yalm in yalms.keys():
    with open(f"./chronos-forecasting/scripts/evaluation/configs/{yalm}.yaml", "w") as file:
        file.write(yalms[yalm])
    print(f"Archivo YAML {yalm} guardado como config.yaml")

################################################################# TRAINING ###############################################################
context_length = 960
prediction_length = 240
max_steps = 15000
save_steps = 1000
per_device_train_batch_size = 16
learning_rate = 0.001
random_init = "true"
shuffle_buffer_length = 10000
output_model = f"/context_{context_length}_pred_{prediction_length}_lr_{learning_rate}"

yaml_content = f"""
training_data_paths:
  - "../../datasets/training/newExchangeRateTraining.arrow"
probability:
  - 1.0
context_length: {context_length}
prediction_length: {prediction_length}
min_past: 60
max_steps: {max_steps}
save_steps: {max_steps}
log_steps: 500
per_device_train_batch_size: {per_device_train_batch_size}
learning_rate: {learning_rate}
optim: adamw_torch_fused
num_samples: 20
shuffle_buffer_length: {shuffle_buffer_length}
gradient_accumulation_steps: 1
model_id: google/t5-efficient-small
model_type: seq2seq
random_init: {random_init}
tie_embeddings: true
output_dir: {output_model}
tf32: true
torch_compile: true
tokenizer_class: "MeanScaleUniformBins"
tokenizer_kwargs:
  low_limit: -15.0
  high_limit: 15.0
n_tokens: 4096
lr_scheduler_type: linear
warmup_ratio: 0.0
dataloader_num_workers: 1
max_missing_prop: 0.9
use_eos_token: true
"""
output_dir = f"chronos-forecasting/scripts/training/configs"
import os
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Guardar el contenido YAML en un archivo
with open(f"{output_dir}/configs_new_dataset_context_{context_length}_pred_{prediction_length}_lr_{learning_rate}.yaml", "w") as file:
    file.write(yaml_content)
print("Archivo YAML guardado como config.yaml")



# Smail Yalm slithy modified
yaml_content = """training_data_paths:
- "../datasets/exchange_rate.arrow"
probability:
- 1
context_length: 512
prediction_length: 64
min_past: 60
max_steps: 200_000
save_steps: 100_000
log_steps: 500
per_device_train_batch_size: 32
learning_rate: 0.001
optim: adamw_torch_fused
num_samples: 20
shuffle_buffer_length: 100_000
gradient_accumulation_steps: 1
model_id: google/t5-efficient-small
model_type: seq2seq
random_init: true
tie_embeddings: true
output_dir: ./output/
tf32: true
torch_compile: true
tokenizer_class: "MeanScaleUniformBins"
tokenizer_kwargs:
  low_limit: -15.0
  high_limit: 15.0
n_tokens: 4096
lr_scheduler_type: linear
warmup_ratio: 0.0
dataloader_num_workers: 1
max_missing_prop: 0.9
use_eos_token: true"""

# Guardar el contenido YAML en un archivo
with open(f"{output_dir}/chronos-t5-small.yaml", "w") as file:
    file.write(yaml_content)
print("Archivo YAML guardado como config.yaml")