#!/usr/bin/env python3
"""Module for converting DataFrame columns to a numpy ndarray"""


def array(df):
    """Selects last 10 rows of High and Close columns as a numpy ndarray"""
    return df[['High', 'Close']].tail(10).to_numpy()
