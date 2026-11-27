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
    if total_good == 0 or total_bad == 0:
        raise ValueError(f"Target must contain both good (0) and bad (1) samples for feature '{feature}'.")

    # Group by the feature bin
    grouped = df.groupby(feature, observed=False)[target].agg(['count', 'sum'])
    grouped = grouped.rename(columns={'count': 'total', 'sum': 'bad_count'})
    grouped['good_count'] = grouped['total'] - grouped['bad_count']

    # Apply Laplace-like smoothing if 0 counts are present to prevent log(0)
    epsilon = 1e-6
    grouped['good_count'] = np.where(grouped['good_count'] == 0, grouped['good_count'] + epsilon, grouped['good_count'])
    grouped['bad_count'] = np.where(grouped['bad_count'] == 0, grouped['bad_count'] + epsilon, grouped['bad_count'])

    # Calculate distributions
    grouped['good_distribution'] = grouped['good_count'] / total_good
    grouped['bad_distribution'] = grouped['bad_count'] / total_bad

    # Calculate WOE and IV
    grouped['WOE'] = np.log(grouped['good_distribution'] / grouped['bad_distribution'])
    grouped['IV'] = (grouped['good_distribution'] - grouped['bad_distribution']) * grouped['WOE']

    result = grouped.reset_index()
    result = result.rename(columns={feature: 'bin'})
    result['bin'] = result['bin'].astype(str)

    return result[['bin', 'good_count', 'bad_count', 'good_distribution', 'bad_distribution', 'WOE', 'IV']]


def transform_to_woe(dataframe, feature, binning_rules):
    """
    Transform a categorical or binned numerical feature to WOE values using the training-derived mapping.
    """
    df = dataframe.copy()
    woe_map = dict(zip(binning_rules['bin'].astype(str), binning_rules['WOE']))
    df[f"{feature}_WOE"] = df[feature].astype(str).map(woe_map).fillna(0.0)

    return df
