

python ../chronos_forecasting/evaluation/evaluate.py evaluation/configs/in-domain.yaml evaluation/results/chronos-t5-large-96-in-domain.csv \
    --chronos-model-id "google_t5-efficient-large/run-0/checkpoint-8000" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20