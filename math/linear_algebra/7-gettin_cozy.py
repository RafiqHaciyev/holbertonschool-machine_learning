#!/usr/bin/env python3
"""Module for concatenating two 2D matrices along a specific axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two 2D matrices along a specific axis.

    Args:
        mat1: A 2D list of ints/floats.
        mat2: A 2D list of ints/floats.
        axis: The axis along which to concatenate (0 or 1).

    Returns:
        A new 2D list with the matrices concatenated, or None if invalid.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]
    return None
