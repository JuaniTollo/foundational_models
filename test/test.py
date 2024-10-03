import unittest
from datasets import load_dataset
import numpy as np
import sys
import os

# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import to_pandas

class TestDataset(unittest.TestCase):

    def setUp(self):
        # Sample data to create a Hugging Face dataset
        self.dataset = load_dataset("autogluon/chronos_datasets", "exchange_rate", split="train")

    def test_timestamp_consistency(self):
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
    
    def test_no_missing_values(self):
        # Check if there are any missing values in the 'target' column
        has_missing_values = self.dataset['target'].count(None) > 0
        self.assertFalse(has_missing_values, "There are missing values in the 'target' column.")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
