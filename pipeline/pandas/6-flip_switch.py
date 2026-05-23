#!/usr/bin/env python3
"""Module for flipping and switching a pandas DataFrame"""


def flip_switch(df):
    """Sorts in reverse chronological order and transposes the DataFrame"""
    return df.sort_index(ascending=False).transpose()
