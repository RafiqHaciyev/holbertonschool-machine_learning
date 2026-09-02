#!/usr/bin/env python3
"""Module that shuffles data points in two matrices the same way."""
import numpy as np


def shuffle_data(X, Y):
    """Shuffle the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): Array of shape (m, nx) to shuffle.
            m is the number of data points.
            nx is the number of features in X.
        Y (numpy.ndarray): Array of shape (m, ny) to shuffle.
            m is the same number of data points as in X.
            ny is the number of features in Y.

    Returns:
        The shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
