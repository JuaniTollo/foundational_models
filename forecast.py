import pandas as pd
import torch
from chronos import ChronosPipeline
from datasets import load_dataset, load_from_disk
from utils import to_pandas
import matplotlib.pyplot as plt
import numpy as np
import pdb

def load_new_dataset():
    dataset = load_dataset('juantollo/newExchangeRate', split='train')
    df = to_pandas(dataset)
    return df.reset_index(drop=True)

def load_and_prepare_data():
    """Load the dataset and prepare the specific currency data."""
    df_new_dataset = load_new_dataset()
    ds_exchange_rate = load_dataset("autogluon/chronos_datasets", "exchange_rate", split="train")
    df_exchange_rate = to_pandas(ds_exchange_rate)
    
    # Return both datasets
    return df_exchange_rate, df_new_dataset

def create_pipeline(model_name="amazon/chronos-t5-large", device="cuda"):
    """Create and return the Chronos pipeline."""
    return ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

def make_forecast(pipeline, context_data, prediction_length=8, num_samples=20):
    """Generate the forecast using the pipeline."""
    forecast = pipeline.predict(
        context=torch.tensor(context_data["target"].values),
        prediction_length=prediction_length,
        num_samples=num_samples,
    )
    return forecast

def plot_forecasts_for_all_currencies(df_exchange_rate, df_new_dataset, pipeline, plot_path):
    """Plot forecasts for all 8 currencies in a single figure with 8 subplots."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))  # 4x2 grid of subplots

    for i in range(8):
        currency_id = f"currency_{i+1}"
        
        # Filter data for the current currency
        df_currency = df_exchange_rate[df_exchange_rate["id"] == currency_id].reset_index(drop=True)
        df_currency_true = df_currency.iloc[-8:]
        df_currency_context = df_currency.iloc[-72:-8].reset_index(drop=True)
        
        # Get the positions mod 8 + i for the new dataset
        positions_mod_8 = np.arange(i, len(df_new_dataset), 8)
        new_forecast = make_forecast(pipeline, df_new_dataset.iloc[positions_mod_8[-576:-64]], prediction_length=64, num_samples=20)
        new_forecast_8_mod = new_forecast[..., ::8]

        # Make the forecast for the historical data
        forecast = make_forecast(pipeline, df_currency_context, prediction_length=8, num_samples=20)

        # Calculate quantiles
        forecast_index = np.arange(len(df_currency_context), len(df_currency_context) + 8)
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        low_new, median_new, high_new = np.quantile(new_forecast_8_mod[0].numpy(), [0.1, 0.5, 0.9], axis=0)

        # Plotting on the appropriate subplot
        ax = axes[i // 2, i % 2]  # Locate the correct subplot
        ax.plot(df_currency_context.index, df_currency_context["target"], color="royalblue", label="historical data")
        ax.plot(forecast_index, median, color="tomato", label="median forecast")
        ax.plot(forecast_index, df_currency_true["target"].values, color="green", label="ground truth")
        ax.plot(forecast_index, median_new, color="purple", label="median new forecast")
        ax.fill_between(forecast_index, low_new, high_new, color="#D8BFD8", alpha=0.3, label="80% prediction interval new")
        ax.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
        
        # Title and legend
        ax.set_title(f"Forecast for {currency_id}")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig(plot_path)

def main():
    # Create the forecasting pipeline
    pipeline = create_pipeline()
    
    # Load and prepare the datasets
    df_exchange_rate, df_new_dataset = load_and_prepare_data()
    
    pdb.set_trace()

    # Plot forecasts for all currencies
    plot_forecasts_for_all_currencies(df_exchange_rate, df_new_dataset, pipeline, "forecasts_all_currencies.png")

if __name__ == "__main__":
    main()
