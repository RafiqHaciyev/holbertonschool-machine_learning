#!/usr/bin/env python3
"""Module for sorting a pandas DataFrame by High price"""


def high(df):
    """Sorts the DataFrame by High price in descending order"""
    return df.sort_values(by='High', ascending=False)
