#!/usr/bin/env python3
"""Module for adding two matrices of any dimension."""


def add_matrices(mat1, mat2):
    """Add two matrices of any dimension element-wise.

    Args:
        mat1: A matrix (nested list) of ints/floats.
        mat2: A matrix (nested list) of ints/floats.

    Returns:
        A new matrix with element-wise sums, or None if shapes differ.
    """
    if isinstance(mat1, list):
        if not isinstance(mat2, list) or len(mat1) != len(mat2):
            return None
        result = [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
        if None in result:
            return None
        return result
    return mat1 + mat2
