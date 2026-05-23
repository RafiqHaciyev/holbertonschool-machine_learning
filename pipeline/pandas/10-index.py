#!/usr/bin/env python3
"""Module for indexing a pandas DataFrame on Timestamp"""


def index(df):
    """Sets the Timestamp column as the index of the DataFrame"""
    return df.set_index('Timestamp')
