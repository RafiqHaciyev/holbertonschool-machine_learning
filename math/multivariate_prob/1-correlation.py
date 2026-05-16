#!/usr/bin/env python3
"""Module for calculating a correlation matrix from a covariance matrix."""
import numpy as np


def correlation(C):
    """Calculate a correlation matrix from a covariance matrix.

    Args:
        C (numpy.ndarray): Array of shape (d, d) containing a covariance
            matrix, where d is the number of dimensions.

    Returns:
        numpy.ndarray: Array of shape (d, d) containing the correlation
            matrix.

    Raises:
        TypeError: If C is not a numpy.ndarray.
        ValueError: If C does not have shape (d, d).
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    corr = C / np.outer(std, std)

    return corr
