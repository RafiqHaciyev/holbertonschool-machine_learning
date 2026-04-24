#!/usr/bin/env python3
"""Module for concatenating two matrices along a specific axis."""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis.

    Args:
        mat1: A nested list of ints/floats.
        mat2: A nested list of ints/floats.
        axis: The axis along which to concatenate (default 0).

    Returns:
        A new matrix with the matrices concatenated, or None if invalid.
    """
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if axis == 0:
        if isinstance(mat1[0], list) != isinstance(mat2[0], list):
            return None
        if isinstance(mat1[0], list) and len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] if isinstance(row, list)
                else row for row in mat1] + \
               [row[:] if isinstance(row, list)
                else row for row in mat2]
    if len(mat1) != len(mat2):
        return None
    result = [cat_matrices(mat1[i], mat2[i], axis - 1)
              for i in range(len(mat1))]
    if None in result:
        return None
    return result
