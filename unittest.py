import unittest
import pandas as pd
from fairness_metrics import calculate_fairness_metrics

class TestFairnessMetrics(unittest.TestCase):
    def test_calculate_fairness_metrics(self):
        # create example input and output dataframes
        input_df = pd.DataFrame({'user_id': [1, 2, 3, 4, 5],
                                'age': [20, 25, 30, 35, 40],
                                'gender': ['male', 'female', 'male', 'female', 'male'],
                                'income': [50000, 55000, 60000, 65000, 70000]})

        output_df = pd.DataFrame({'user_id': [1, 2, 3, 4, 5],
                                 'model_result': [1, 0, 1, 1, 0],
                                 'actual': [1, 1, 0, 0, 1]})

        # calculate fairness metrics
        metrics = calculate_fairness_metrics(input_df, output_df, 'gender', 'female')

        # check that the returned object is a pandas dataframe
        self.assertIsInstance(metrics, pd.DataFrame)

        # check that the returned dataframe has the expected columns
        self.assertIn('metric', metrics.columns)
        self.assertIn('protected', metrics.columns)
        self.assertIn('unprotected', metrics.columns)

if __name__ == '__main__':
    unittest.main()
