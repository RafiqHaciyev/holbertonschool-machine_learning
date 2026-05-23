#!/usr/bin/env python3
import numpy as np
import pandas as pd


def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray"""
    import string
    cols = list(string.ascii_uppercase[:array.shape[1]])
    return pd.DataFrame(array, columns=cols)
