#!/usr/bin/env python3
"""Module for renaming and converting DataFrame columns"""
import pandas as pd


def rename(df):
    """Renames Timestamp column to Datetime, converts values, returns Datetime and Close"""
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
