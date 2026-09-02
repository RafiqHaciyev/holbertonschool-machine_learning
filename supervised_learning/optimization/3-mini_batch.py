#!/usr/bin/env python3
"""Module that creates mini-batches for mini-batch gradient descent."""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create mini-batches to be used for training a neural network
    using mini-batch gradient descent.

    Args:
        X (numpy.ndarray): Array of shape (m, nx) representing input
            data.
            m is the number of data points.
            nx is the number of features in X.
        Y (numpy.ndarray): Array of shape (m, ny) representing the
            labels.
            m is the same number of data points as in X.
            ny is the number of classes for classification tasks.
        batch_size: the number of data points in a batch.

    Returns:
        list: mini-batches containing tuples (X_batch, Y_batch).
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
