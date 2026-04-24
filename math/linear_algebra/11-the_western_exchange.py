#!/usr/bin/env python3
"""Module for transposing a numpy.ndarray."""


def np_transpose(matrix):
    """Transpose a numpy.ndarray.

    Args:
        matrix: A numpy.ndarray.

    Returns:
        A new numpy.ndarray representing the transposed matrix.
    """
    return matrix.T.copy()
