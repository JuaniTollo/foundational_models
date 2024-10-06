import pandas as pd
import torch
from chronos import ChronosPipeline
from datasets import load_dataset
from utils import to_pandas
import matplotlib.pyplot as plt
import numpy as np

def load_new_dataset():
    dataset = load_dataset('juantollo/newExchangeRate', split='train')
    df = to_pandas(dataset)
    return df.reset_index(drop=True)

def load_and_prepare_data():
    """Load the dataset and prepare the specific currency data."""
    df_new_dataset = load_new_dataset()
    ds_exchange_rate = load_dataset("autogluon/chronos_datasets", "exchange_rate", split="train")
    df_exchange_rate = to_pandas(ds_exchange_rate)
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

# Min-Max Normalization Function with parameter storage
def min_max_normalize(series, series_id, normalization_params, range_min=-15, range_max=15):
    min_val = series.min()
    max_val = series.max()
    # Store normalization parameters
    normalization_params[series_id] = {'min': min_val, 'max': max_val}
    # Normalize to the range [-15, 15]
    normalized = (series - min_val) / (max_val - min_val)
    scaled = normalized * (range_max - range_min) + range_min
    return scaled

# Min-Max Denormalization Function using stored parameters
def min_max_denormalize(series, series_id, normalization_params, range_min=-15, range_max=15):
    # Check if parameters are stored
    if series_id not in normalization_params:
        print(f"Normalization parameters for {series_id} not found in min_max_denormalize.")
        raise KeyError(f"Normalization parameters for {series_id} not found.")
    params = normalization_params[series_id]
    min_val = params['min']
    max_val = params['max']
    # Reverse scaling to the range [0, 1]
    normalized = (series - range_min) / (range_max - range_min)
    # Denormalize back to the original range
    denormalized = normalized * (max_val - min_val) + min_val
    return denormalized

# Z-Score Normalization Function with parameter storage
def z_score_normalize(series, series_id, normalization_params, range_min=-5, range_max=5):
    mean = series.mean()
    std_dev = series.std()
    # Store normalization parameters
    normalization_params[series_id] = {'mean': mean, 'std_dev': std_dev}
    # Apply the z-score normalization
    normalized = (series - mean) / std_dev
    # Scale to the desired range [-15, 15]
    max_abs_value = max(abs(normalized.min()), abs(normalized.max()))
    if max_abs_value == 0:
        scaling_factor = 1  # Avoid division by zero
    else:
        scaling_factor = (range_max - range_min) / (2 * max_abs_value)
    # Store scaling factor
    normalization_params[series_id]['scaling_factor'] = scaling_factor
    scaled = normalized * scaling_factor
    return scaled

# Z-Score Denormalization Function using stored parameters
def z_score_denormalize(series, series_id, normalization_params):
    # Check if parameters are stored
    if series_id not in normalization_params:
        print(f"Normalization parameters for {series_id} not found in z_score_denormalize.")
        raise KeyError(f"Normalization parameters for {series_id} not found.")
    params = normalization_params[series_id]
    mean = params['mean']
    std_dev = params['std_dev']
    scaling_factor = params.get('scaling_factor', 1)
    # Reverse scaling
    denormalized = series / scaling_factor
    # Reverse z-score normalization
    denormalized = denormalized * std_dev + mean
    return denormalized

# Apply normalization to each mod-8 series with parameter storage
def normalize_by_mod_8(df, target_column, normalization_func, normalization_params):
    normalized_df = df.copy()
    for i in range(8):
        # Select the positions mod 8
        positions_mod_8 = np.arange(i, len(df), 8)
        # Filter indices to include only those within the DataFrame length
        valid_indices = positions_mod_8[positions_mod_8 < len(df)]
        # Use normalization function and store parameters using series_id
        series_id = f'mod_8_series_{i}'
        series_data = df.iloc[valid_indices][target_column]
        normalized_series = normalization_func(series_data, series_id, normalization_params)
        normalized_df.iloc[valid_indices, normalized_df.columns.get_loc(target_column)] = normalized_series.values
    return normalized_df

# Modify the plotting function to use series_id for denormalization
def plot_forecasts_for_all_currencies(df_exchange_rate, df_new_dataset, pipeline, plot_path, normalization_params, normalize="df_normalized_min_max"):
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
        positions_mod_8 = positions_mod_8[positions_mod_8 < len(df_new_dataset)]  # Ensure indices are valid
        context_positions = positions_mod_8[-576:-64]
        if len(context_positions) == 0:
            continue  # Skip if there is not enough data
        new_forecast = make_forecast(pipeline, df_new_dataset.iloc[context_positions], prediction_length=64, num_samples=20)
        new_forecast_8_mod = new_forecast[..., ::8]

        # Make the forecast for the historical data
        forecast = make_forecast(pipeline, df_currency_context, prediction_length=8, num_samples=20)

        # Denormalize forecasts if needed using series_id
        series_id = f'mod_8_series_{i}'
        print(f"Attempting to denormalize for {series_id}")
        if normalize == "df_normalized_min_max":
            new_forecast_values = new_forecast_8_mod[0].numpy()
            denormalized_new_forecast = min_max_denormalize(new_forecast_values, series_id, normalization_params)
        elif normalize == "df_normalized_z_score":
            new_forecast_values = new_forecast_8_mod[0].numpy()
            denormalized_new_forecast = z_score_denormalize(new_forecast_values, series_id, normalization_params)
        else:
            denormalized_new_forecast = new_forecast_8_mod[0].numpy()

        # For the historical data forecast, no denormalization is needed
        forecast_values = forecast[0].numpy()

        # Calculate quantiles
        forecast_index = np.arange(len(df_currency_context), len(df_currency_context) + 8)
        low, median, high = np.quantile(forecast_values, [0.1, 0.5, 0.9], axis=0)
        low_new, median_new, high_new = np.quantile(denormalized_new_forecast, [0.1, 0.5, 0.9], axis=0)

        # Plotting on the appropriate subplot
        ax = axes[i // 2, i % 2]
        ax.plot(df_currency_context.index, df_currency_context["target"], color="royalblue", label="historical data")
        ax.plot(forecast_index, median, color="tomato", label="median forecast")
        ax.plot(forecast_index, df_currency_true["target"].values, color="green", label="ground truth")
        ax.plot(forecast_index, median_new[:8], color="purple", label="median new forecast")
        ax.fill_between(forecast_index, low_new[:8], high_new[:8], color="#D8BFD8", alpha=0.3, label="80% prediction interval new")
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

    # Choose which normalization method to use for plotting
    #normalization_method = "df_normalized_min_max"  # or "df_normalized_z_score"
    normalization_method = "df_normalized_z_score"  # or "df_normalized_z_score"

    if normalization_method == "df_normalized_min_max":
        normalization_params = {}
        df_normalized = normalize_by_mod_8(df_new_dataset, 'target', min_max_normalize, normalization_params)
    elif normalization_method == "df_normalized_z_score":
        normalization_params = {}
        df_normalized = normalize_by_mod_8(df_new_dataset, 'target', z_score_normalize, normalization_params)
    else:
        normalization_params = {}
        df_normalized = df_new_dataset  # No normalization

    # Plot forecasts for all currencies
    plot_forecasts_for_all_currencies(df_exchange_rate,
                                      df_normalized,
                                      pipeline, 
                                      f"forecasts_all_currencies_{normalization_method}.png", 
                                      normalization_params,
                                      normalize=normalization_method)

if __name__ == "__main__":
    main()
