#!/usr/bin/env python3
"""Module that calculates normalization constants of a matrix."""
import numpy as np


def normalization_constants(X):
    """Calculate the normalization (standardization) constants of a matrix.

    Args:
        X (numpy.ndarray): Array of shape (m, nx) to normalize.
            m is the number of data points.
            nx is the number of features.

    Returns:
        mean, std: the mean and standard deviation of each feature,
            respectively.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean, std
