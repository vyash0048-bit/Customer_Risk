import pandas as pd
import numpy as np

def calculate_woe_iv(dataframe, feature, target):
    """
    Calculate Weight of Evidence (WOE) and Information Value (IV) for a feature.
    """
    df = dataframe[[feature, target]].copy()
    
    # Total good (0) and bad (1) counts
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()
    
    # Group by the feature bin
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped = grouped.rename(columns={'count': 'total', 'sum': 'bad_count'})
    grouped['good_count'] = grouped['total'] - grouped['bad_count']
    
    # Apply smoothing if 0 counts are present to prevent log(0)
    epsilon = 1e-6
    grouped['good_count'] = np.where(grouped['good_count'] == 0, grouped['good_count'] + epsilon, grouped['good_count'])
    grouped['bad_count'] = np.where(grouped['bad_count'] == 0, grouped['bad_count'] + epsilon, grouped['bad_count'])
    
    # Need to adjust totals slightly if smoothing was applied, but usually standard is to just use adjusted counts 
    # and original totals, or adjusted totals. Here we use original total for distribution.
    
    # Calculate distributions
    grouped['good_distribution'] = grouped['good_count'] / total_good
    grouped['bad_distribution'] = grouped['bad_count'] / total_bad
    
    # Calculate WOE and IV
    grouped['WOE'] = np.log(grouped['good_distribution'] / grouped['bad_distribution'])
    grouped['IV'] = (grouped['good_distribution'] - grouped['bad_distribution']) * grouped['WOE']
    
    # Format output to match requested structure
    result = grouped.reset_index()
    result = result.rename(columns={feature: 'bin'})
    
    return result[['bin', 'good_count', 'bad_count', 'good_distribution', 'bad_distribution', 'WOE', 'IV']]

def transform_to_woe(dataframe, feature, binning_rules):
    """
    Transform a categorical or binned numerical feature to WOE values using the training-derived mapping.
    
    Parameters:
    - dataframe: pandas DataFrame to transform
    - feature: string name of the column to transform
    - binning_rules: pandas DataFrame resulting from calculate_woe_iv
    
    Returns:
    - Transformed DataFrame with a new '{feature}_WOE' column
    """
    df = dataframe.copy()
    
    woe_map = dict(zip(binning_rules['bin'], binning_rules['WOE']))
    df[f"{feature}_WOE"] = df[feature].map(woe_map).fillna(0.0)
    
    return df
