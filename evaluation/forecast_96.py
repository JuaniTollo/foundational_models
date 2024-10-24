import pandas as pd
import torch
from chronos import ChronosPipeline
from datasets import load_dataset, load_from_disk
from utils.utils import to_pandas
import matplotlib.pyplot as plt
import numpy as np
import pdb

n = 8
t = 96

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

def plot_forecasts_for_all_currencies(df_exchange_rate, df_new_dataset, pipeline_original_model, pipeline_new_model, plot_path):
    """Plot forecasts for all 8 currencies in a single figure with 8 subplots."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))  # 4x2 grid of subplots

    for i in range(8):
        currency_id = f"currency_{i+1}"
        
        # Filter data for the current currency
        df_currency = df_exchange_rate[df_exchange_rate["id"] == currency_id].reset_index(drop=True)
        df_currency_true = df_currency.iloc[-t:]
        df_currency_context = df_currency.iloc[-t*n:-t].reset_index(drop=True)
        
        # Get the positions mod 8 + i for the new dataset
        positions_mod_n = np.arange(i, n*t, n)
        new_forecast = make_forecast(pipeline_new_model, df_new_dataset.iloc[-516*n:-t*n], prediction_length=t*n, num_samples=20)
  
        new_forecast_n_mod = new_forecast[:,:,positions_mod_n]

        # Make the forecast for the historical data
        forecast = make_forecast(pipeline_original_model, df_currency_context, prediction_length=30, num_samples=20)

        # Calculate quantiles
        forecast_index = np.arange(len(df_currency_context), len(df_currency_context) + 30)
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        low_new, median_new, high_new = np.quantile(new_forecast_n_mod[0].numpy(), [0.1, 0.5, 0.9], axis=0)

        # Plotting on the appropriate subplot
        ax = axes[i // 2, i % 2]  # Locate the correct subplot

        combined_index = np.concatenate([df_currency_context.index, forecast_index])
        combined_target = np.concatenate([df_currency_context["target"].values, df_currency_true["target"].values])

        # Plot the combined data
        ax.plot(combined_index, combined_target, color="royalblue", label="historical data + ground truth")
        ax.plot(forecast_index, median, color="tomato", label="median forecast")
        ax.plot(forecast_index, median_new, color="purple", label="median new forecast")
        ax.fill_between(forecast_index, low_new, high_new, color="purple", alpha=0.3, label="80% prediction interval new")
        ax.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
        
        # Title and legend
        ax.set_title(f"Forecast for {currency_id}")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig(plot_path)

def main(new_model_path=None):
    # Create the forecasting pipeline
    pipeline_original_model = create_pipeline(original_model_path, "cuda")
    
    # Check if a new model path is provided, then create a pipeline for it
    if new_model_path:
        pipeline_new_model = create_pipeline(new_model_path, "cuda")
    else:
        pipeline_new_model = None

    # Load and prepare the datasets
    df_exchange_rate, df_new_dataset = load_and_prepare_data()
    
    # Plot forecasts for all currencies
    plot_forecasts_for_all_currencies(df_exchange_rate, df_new_dataset, pipeline_original_model, pipeline_new_model, "plots/forecasts_all_currencies_fine_tuned.png")
import argparse

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Forecasting script")
    parser.add_argument('original_model_path', type=str, help='Path to the original model')
    parser.add_argument('new_model_path', type=str, help='Path to the new model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Pass the new_model_path argument to main
    main(new_model_path=args.new_model_path, original_model_path=args.original_model_path)