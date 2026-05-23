#!/usr/bin/env python3
"""Module for concatenating two pandas DataFrames"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Indexes DataFrames on Timestamp, filters df2, and concatenates them"""
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2[df2.index <= 1417411920]
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
