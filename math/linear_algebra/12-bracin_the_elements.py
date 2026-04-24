#!/usr/bin/env python3
"""Module for element-wise operations on numpy.ndarrays."""


def np_elementwise(mat1, mat2):
    """Perform element-wise add, sub, mul, and div.

    Args:
        mat1: A numpy.ndarray.
        mat2: A numpy.ndarray or scalar.

    Returns:
        A tuple of (sum, difference, product, quotient).
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
