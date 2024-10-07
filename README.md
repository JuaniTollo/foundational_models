## Create new datasets and configs files

We import exchange rate dataset and we create new exchange rante datasets

`python create_excghange_rate_datasets.py`

## Evaluation for the original dataset with the original model weights

python evaluation/evaluate.py evaluation/configs/exchange_zero_shot_test.yaml evaluation/results/chronos-t5-small-zero-shotORIGINAL.csv     --chronos-model-id "amazon/chronos-t5-small"     --batch-size=32     --device=cuda:0     --num-samples 20

## Evaluation for new dataset but original weights

python evaluation/evaluate.py /home/juantollo/foundational_models/chronos-forecasting/scripts/evaluation/configs/new_exchange_rate_zero_shot.yaml evaluation/results/chronos-t5-small-zero-shot.csv     --chronos-model-id "/home/juantollo/foundational_models/chronos-forecasting/scripts/er_models/5000"    --batch-size=3

## VM Settings

```python
pip install -r requirements.txt

pip install --upgrade --no-deps --ignore-installed -r requirements.txt

pip install -e .[evaluation,training]
```

## Tests Evaluate

python evaluation/evaluate.py evaluation/configs/exchange_zero_shot_test.yaml evaluation/results/chronos-t5-small-zero-shot_test.csv --chronos-model-id amazon/chronos-t5-small --batch-size 2 --device cuda:0 --num-samples 1

## Test train new dataset

```
## `CUDA_VISIBLE_DEVICES=0 python training/train.py --config /home/juantollo/foundational_models/chronos-forecasting/scripts/training/configs/configs_new_dataset_context_960_pred_240_lr_0.001.yaml     --model-id amazon/chronos-t5-small     --no-random-init     --max-steps 1000     --learning-rate 0.001`
```

## Test train original dataset

```sh
# Fine-tune `amazon/chronos-t5-small` for 1000 steps with initial learning rate of 1e-3
CUDA_VISIBLE_DEVICES=0 python training/train.py --config /home/juantollo/foundational_models/chronos-forecasting/scripts/training/configs/chronos-t5-small.yaml \
    --model-id amazon/chronos-t5-small \
    --no-random-init \
    --max-steps 1000 \
    --learning-rate 0.001
```
