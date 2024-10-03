new_exchange_rate_zero_shot = """
- name: exchange_rate
  hf_repo: autogluon/chronos_datasets
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

yalms = {"new_exchange_rate_zero_shot": new_exchange_rate_zero_shot,
         "exchange_zero_shot": exchange_zero_shot
         } 

for yalm in yalms.keys():
    with open(f"./chronos-forecasting/scripts/evaluation/configs/{yalm}_zero_shot.yaml", "w") as file:
        file.write(yalms[yalm])
