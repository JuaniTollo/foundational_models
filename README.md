# Setting Up Chronos Forecasting Environment

Follow these steps to clone the repositories, install the dependencies, and run the training scripts.

## Step 1: Clone the Repositories

Make sure you're at the same directory level as `foundational_models` before cloning:
Clone the official Chronos Forecasting repository
```
git clone https://github.com/amazon-science/chronos-forecasting.git
```
Clone your custom branch of Chronos Forecasting
```
git clone --branch new-train https://github.com/JuaniTollo/chronos-forecasting.git /content/chronos-forecasting --quiet
```

## Step 2: Install Dependencies

Install the dependencies from the requirements file
```
pip install -r requirements.txt
```
Upgrade the dependencies, ignoring any pre-installed packages
```
pip install --upgrade --no-deps --ignore-installed -r requirements.txt
```
## Step 3: Install Chronos in Editable Mode
Navigate into the chronos-forecasting directory and install it in editable mode
```
cd ./chronos-forecasting
```
Install Chronos with both evaluation and training extras
```
pip install -e .[evaluation,training]
```
## Step 4: Run Configuration and Dataset Script
Run the script to create the necessary configurations and datasets

```
run /home/juantollo/foundational_models/foundational_models/create_config_and_datasets.sh
```
## Step 5: Run the Model
Now, you can run the model by specifying the model size you prefer (tiny, small, base, or large)

```
run /home/juantollo/foundational_models/foundational_models/run.sh large
```