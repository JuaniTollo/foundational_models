#!/bin/bash

# Source the .env file to set HUGGINGFACE_TOKEN
source .env

# Log in to Hugging Face using the token from the environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN


# python ../chronos-forecasting/scripts/evaluation/evaluate.py \
#     evaluation/configs/exchange_zero_zero_shot_w96.yaml \
#     evaluation/results/chronos-t5-large-96-zero-shot_new_model.csv \
#     --chronos-model-id "amazon/chronos-t5-large" \
#     --batch-size=32 \
#     --device=cuda:0 \
#     --num-samples 20

# python ../chronos-forecasting/scripts/evaluation/evaluate.py \
#     evaluation/configs/new_exchange_rate_in_domain.yaml \
#     evaluation/results/chronos-t5-large-96-in-domain.csv \
#     --chronos-model-id "juantollo/checkpoint-final_20241009-225937" \
#     --batch-size=32 \
#     --device=cuda:0 \
#     --num-samples 20

echo "Script execution completed."

# Consolidate the CSS files into one aggregated CSV file
python evaluation/consolidate_csv.py
echo "Script execution completed."

python evaluation/forecast_96.py "juantollo/checkpoint-final_20241009-225937" "amazon/chronos-t5-large"
echo "Script forecast completed."
