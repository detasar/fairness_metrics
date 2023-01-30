import pandas as pd
import numpy as np

def fairness_metrics(input_df, output_df, protected_feature, protected_value):
    """
    Calculates fairness metrics for a binary classification model based on protected and unprotected population.

    Parameters
    ----------
    input_df : pandas dataframe
        The input dataframe that includes user features.
    output_df : pandas dataframe
        The output dataframe that includes user_id and model results column of a binary classification case.
    protected_feature : str
        The name of the column in input_df that includes protected feature values.
    protected_value : str
        The protected value in the protected_feature column to calculate metrics for.

    Returns
    -------
    pandas dataframe
        A dataframe that includes all calculated fairness metrics.
    """
    # merge the two dataframes based on user_id
    merged_df = pd.merge(input_df, output_df, on='user_id')
    
    # separate protected and unprotected population
    protected_pop = merged_df[merged_df[protected_feature] == protected_value]
    unprotected_pop = merged_df[merged_df[protected_feature] != protected_value]
    
    # calculate equal parity
    protected_eq_parity = sum(protected_pop['model_result']) / len(protected_pop)
    unprotected_eq_parity = sum(unprotected_pop['model_result']) / len(unprotected_pop)
    eq_parity = protected_eq_parity - unprotected_eq_parity
    
    # calculate proportional parity
    protected_prop_parity = sum(protected_pop['model_result']) / sum(protected_pop['model_result'] + unprotected_pop['model_result'])
    unprotected_prop_parity = sum(unprotected_pop['model_result']) / sum(protected_pop['model_result'] + unprotected_pop['model_result'])
    prop_parity = protected_prop_parity - unprotected_prop_parity
    
    # calculate false discovery rate parity
    protected_fdr = sum((protected_pop['model_result'] == 1) & (protected_pop['model_result'] != protected_pop['actual'])) / sum(protected_pop['model_result'] == 1)
    unprotected_fdr = sum((unprotected_pop['model_result'] == 1) & (unprotected_pop['model_result'] != unprotected_pop['actual'])) / sum(unprotected_pop['model_result'] == 1)
    fdr_parity = protected_fdr - unprotected_fdr
    
    # calculate false positive rate parity
    protected_fpr = sum((protected_pop['model_result'] == 1) & (protected_pop['actual'] == 0)) / sum(protected_pop['actual'] == 0)
    unprotected_fpr = sum((unprotected_pop['model_result'] == 1) & (unprotected_pop['actual'] == 0)) / sum(unprotected_pop['actual'] == 0)
    fpr_parity = protected_fpr - unprotected_fpr
    
    # calculate false omission rate parity
    protected_for = sum((protected_group['model_result'] == 0) & (protected_group['actual'] == 1)) / len(protected_group[protected_group['actual'] == 1])
    unprotected_for = sum((unprotected_group['model_result'] == 0) & (unprotected_group['actual'] == 1)) / len(unprotected_group[protected_group['actual'] == 1])
    for_parity = protected_for - unprotected_for

    # calculate false negative rate parity
    protected_fnr = sum((protected_pop['model_result'] == 0) & (protected_pop['actual'] == 1)) / sum(protected_pop['actual'] == 1)
    unprotected_fnr = sum((unprotected_pop['model_result'] == 0) & (unprotected_pop['actual'] == 1)) / sum(unprotected_pop['actual'] == 1)
    fnr_parity = protected_fnr - unprotected_fnr
    
    # calculate recall parity
    protected_recall = sum((protected_pop['model_result'] == 1) & (protected_pop['actual'] == 1)) / sum(protected_pop['actual'] == 1)
    unprotected_recall = sum((unprotected_pop['model_result'] == 1) & (unprotected_pop['actual'] == 1)) / sum(unprotected_pop['actual'] == 1)
    recall_parity = protected_recall - unprotected_recall
    
    # store all calculated metrics in a dictionary
    metrics = {
        'Equal Parity': eq_parity,
        'Proportional Parity': prop_parity,
        'False Discovery Rate Parity': fdr_parity,
        'False Positive Rate Parity': fpr_parity,
        'False Omission Rate Parity': for_parity,
        'False Negative Rate Parity': fnr_parity,
        'Recall Parity': recall_parity
    }
    
    # return the dictionary as a pandas dataframe
    return pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
