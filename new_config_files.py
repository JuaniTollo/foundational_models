################################################################# EVALUATION ###############################################################

new_exchange_rate_zero_shot = """
- name: 'default'
  hf_repo: juantollo/newExchangeRate
  offset: -64
  prediction_length: 64
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

import os
import glob

################################################################ TRAINING ###############################################################
def erased_not_experiment_yammls(output_dir):
    # Find all .yaml files in the specified directory
    yaml_files = glob.glob(os.path.join(output_dir, "*.yaml"))

    # Delete each .yaml file
    for file in yaml_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def write_experiments_yalms(a_model_list, a_lr_list, output_dir):
    for a_model in a_model_list:
        for a_lr in a_lr_list:
            context_length = 960
            prediction_length = 64
            max_steps = 5000
            save_steps = 1000
            per_device_train_batch_size = 16
            learning_rate = a_lr
            random_init = "false"
            shuffle_buffer_length = 5000
            model = a_model
            #output_model = f"/{model}_lr_{learning_rate}_random_init{random_init}"
            output_model = f"{model.replace('/', '_')}_lr_{learning_rate}"


            yaml_content = f"""
training_data_paths:
  - "../../datasets/training/newExchangeRateTraining.arrow"
probability:
  - 1.0
context_length: {context_length}
prediction_length: {prediction_length}
min_past: 60
max_steps: {max_steps}
save_steps: {save_steps}
log_steps: 100
per_device_train_batch_size: {per_device_train_batch_size}
learning_rate: {learning_rate}
optim: adamw_torch_fused
num_samples: 20
shuffle_buffer_length: {shuffle_buffer_length}
gradient_accumulation_steps: 1
model_id: {model}
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
warmup_ratio: 0.1
dataloader_num_workers: 1
max_missing_prop: 0.9
use_eos_token: true
"""
            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Save the YAML content to a file
            yaml_file_path = os.path.join(output_dir, f"{output_model}.yaml")
            with open(yaml_file_path, "w") as file:
                file.write(yaml_content)
            print(f"YAML file saved as {yaml_file_path}")

def main():
    # Define the directory containing the .yaml files
    output_dir = "chronos-forecasting/scripts/training/configs"
    # Erase old YAMLs if necessary
    erased_not_experiment_yammls(output_dir)
    # Write new YAML files
    write_experiments_yalms(["google/t5-efficient-base", "google/t5-efficient-small"], [1e-3, 1e-4, 3e-4], output_dir)

if __name__ == "__main__":
    main()
