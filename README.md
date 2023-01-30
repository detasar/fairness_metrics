# Fairness_metrics

## Fairness Metrics Library for Binary Classification Models
A library for calculating fairness metrics for binary classification models. The library takes as input a dataframe with user features, a dataframe with model outputs, a column name for the protected feature, and the protected value. The library returns a pandas dataframe with fairness metrics, including equal parity, proportional parity, false discovery rate parity, false positive rate parity, false omission rate parity, false negative rate parity, and recall parity.

## Installation
Use the package manager pip to install the library.

```bash

pip install fairness-metrics

```
##Usage

```python
import pandas as pd
from fairness_metrics import calculate_fairness_metrics

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

# print the results
print(metrics)
```

##Example Notebook
```python
import pandas as pd
from fairness_metrics import calculate_fairness_metrics

# create example input and output dataframes
input_df = pd.DataFrame({'user_id': [1, 2, 3, 4, 5],
                        'age': [20, 25, 30, 35, 40],
                        'gender': ['male', 'female', 'male', 'female', 'male'],
                        'income': [50000, 55000, 60000, 65000, 70000]})

output_df = pd.DataFrame({'user_id': [1, 2, 3, 4, 5],
                         'model_result': [1, 0, 1, 1, 0],
                         'actual': [1, 1, 0, 0, 1]})

# merge input and output dataframes on 'user_id'
merged_df = pd.merge(input_df, output_df, on='user_id')

# calculate fairness metrics for protected value 'female'
metrics = calculate_fairness_metrics(merged_df, 'gender', 'female')

# print the resulting metrics dataframe
print(metrics)
```
##Notebook Outputs
```python
           metric  protected  unprotected
0   equal_parity      0.50        0.50
1  prop_parity        0.5        0.5
2  fdr_parity         0.5        0.5
3  fpr_parity         0.5        0.5
4  for_parity         0.0        1.0
5  fnr_parity         1.0        0.0
6    recall_parity    0.5        0.5
```


