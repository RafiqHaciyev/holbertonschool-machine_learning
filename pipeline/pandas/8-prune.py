#!/usr/bin/env python3
"""Module for pruning missing values from a pandas DataFrame"""


def prune(df):
    """Removes rows where Close column has NaN values"""
    return df.dropna(subset=['Close'])
