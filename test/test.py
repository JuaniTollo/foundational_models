import unittest
from datasets import load_dataset
import numpy as np
import sys
import os
import subprocess

# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import to_pandas
import pdb

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Sample data to create a Hugging Face dataset
        self.dataset = load_dataset("autogluon/chronos_datasets", "exchange_rate", split="train")
        self.newdataset = load_dataset("juantollo/newExchangeRate", split="train")
        self.dataset_df = to_pandas(self.dataset)
        self.newdataset_df = to_pandas(self.newdataset)

    def test01_timestamp_consistency(self):
        # Convert dataset to a DataFrame for easier processing
        df = self.dataset.to_pandas()
        # Group by 'id' to get timestamps for each series
        timestamp_groups = df.groupby('id')['timestamp'].apply(list)
        
        # Get the first group's timestamps as the reference
        reference_timestamps = timestamp_groups.iloc[0]

        # Check if all groups have the same timestamps
        for timestamps in timestamp_groups:
        #    self.assertEqual(timestamps, reference_timestamps, "Timestamps are not consistent across series.")
            self.assertTrue(all(np.array_equal(t, reference_timestamps[0]) for t in timestamps), "Timestamps are not consistent across series.")
        
        print("All datasets have consistent timestamps.")
    
    def test02_equal_transformation(self):
        #pdb.set_trace()
        for i in range (0,8):
            #df = self.dataset_df[self.dataset_df["id"] == f"currency_{1}"]["target"].values
            currency_id = f"currency_{i+1}"
            # Filter data for the current currency in the original dataset
            original_currency_values = self.dataset_df[self.dataset_df["id"] == currency_id]["target"].values

            # Determine positions in the new dataset based on mod 8 logic
            positions_mod_8 = np.arange(i, len(self.newdataset_df), 8)
            
            # Extract values from the new dataset at these positions
            new_dataset_values_mod_8 = self.newdataset_df.iloc[positions_mod_8]["target"].values
            
            # Compare the values in the new dataset mod 8 positions with the original dataset
            # Assertion for values at mod 8 positions
            assert np.array_equal(
                new_dataset_values_mod_8, 
                original_currency_values[-len(new_dataset_values_mod_8):]
            ), f"Mismatch found for Currency {i+1} at mod 8 positions."

    def test03_evaluate_exchange_rate(self):
        # Add the path to the scripts directory to the system path
        sys.path.append(os.path.abspath('../chronos-forecasting/scripts'))
        # Optionally, change the working directory to the target directory
        os.chdir(os.path.abspath('../chronos-forecasting/scripts'))

        # Define the command to run the Python file
        command = [
            'python', 'evaluation/evaluate.py', 
            'evaluation/configs/exchange_zero_shot_test.yaml', 
            'evaluation/results/chronos-t5-small-zero-shot_test.csv',
            '--chronos-model-id', 'amazon/chronos-t5-small',
            '--batch-size=2',
            '--device=cuda:0',
            '--num-samples', '1'
        ]
        
        # Run the command and check for errors
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            # Check the return code (implicitly done by check=True)
            self.assertEqual(result.returncode, 0, "The process did not exit successfully.")
            
            # Optionally, check for specific content in the stdout or stderr
            output_content = result.stdout
            if "expected_output_message" not in output_content:
                self.fail("The expected output was not found in the script's output.")
            
            # Check if the output file was created
            output_file = 'evaluation/results/chronos-t5-small-zero-shot_test.csv'
            self.assertTrue(os.path.exists(output_file), "The output file was not created.")
            
        except subprocess.CalledProcessError as e:
            self.fail(f"Evaluation script failed with error: {e}\n{e.stderr}")

if __name__ == '__main__':
    unittest.main(argv=['', 'TestDataset.test02_equal_transformation'], exit=False)