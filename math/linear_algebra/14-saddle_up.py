#!/usr/bin/env python3
"""Module for performing matrix multiplication on numpy.ndarrays."""
import numpy as np


def np_matmul(mat1, mat2):
    """Perform matrix multiplication.

    Args:
        mat1: A numpy.ndarray.
        mat2: A numpy.ndarray.

    Returns:
        A numpy.ndarray representing the matrix product.
    """
    return np.matmul(mat1, mat2)
