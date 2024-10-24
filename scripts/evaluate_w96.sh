#!/bin/bash

# Source the .env file to set HUGGINGFACE_TOKEN
source .env

# Log in to Hugging Face using the token from the environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN


# python ../chronos-forecasting/scripts/evaluation/evaluate.py \
#     evaluation/configs/exchange_zero_zero_shot_w96.yaml \
#     evaluation/results/chronos-t5-large-96-zero-shotRocco.csv \

python ../chronos-forecasting/scripts/evaluation/evaluate.py \
    evaluation/configs/exchange_zero_zero_shot_w96.yaml \
    evaluation/results/chronos-t5-large-96-zero-shot_new_model.csv \

echo "Script execution completed. Check output.log for details."
