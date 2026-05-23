#!/usr/bin/env python3
"""Module for slicing a pandas DataFrame"""


def slice(df):
    """Extracts High, Low, Close, Volume_BTC columns every 60th row"""
    return df[['High', 'Low', 'Close', 'Volume_BTC']][::60]
