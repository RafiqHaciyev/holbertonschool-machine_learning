#!/usr/bin/env python3
"""Module for analyzing a pandas DataFrame"""


def analyze(df):
    """Computes descriptive statistics for all columns except Timestamp"""
    if 'Timestamp' in df.columns:
        df = df.drop(columns='Timestamp')
    return df.describe()
