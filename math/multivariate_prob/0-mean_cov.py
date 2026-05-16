#!/usr/bin/env python3
"""Module for calculating the mean and covariance of a data set."""
import numpy as np


def mean_cov(X):
    """Calculate the mean and covariance of a data set.

    Args:
        X (numpy.ndarray): Array of shape (n, d) containing the data set,
            where n is the number of data points and d is the number of
            dimensions.

    Returns:
        tuple: (mean, cov) where:
            mean (numpy.ndarray): Shape (1, d), mean of the data set.
            cov (numpy.ndarray): Shape (d, d), covariance matrix.

    Raises:
        TypeError: If X is not a 2D numpy.ndarray.
        ValueError: If n is less than 2.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.sum(X, axis=0, keepdims=True) / n
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
