#!/usr/bin/env python3
"""Module for performing matrix multiplication."""


def mat_mul(mat1, mat2):
    """Perform matrix multiplication.

    Args:
        mat1: A 2D list of ints/floats.
        mat2: A 2D list of ints/floats.

    Returns:
        A new 2D list representing the product, or None if invalid.
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
             for j in range(len(mat2[0]))]
            for i in range(len(mat1))]
