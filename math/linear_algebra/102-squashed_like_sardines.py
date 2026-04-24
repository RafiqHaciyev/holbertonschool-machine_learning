#!/usr/bin/env python3
"""Module for concatenating two matrices along a specific axis."""


def get_shape(matrix):
    """Get the shape of a nested list matrix.

    Args:
        matrix: A nested list of ints/floats.

    Returns:
        A list of integers representing the shape.
    """
    shape = []
    current = matrix
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0]
    return shape


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis.

    Args:
        mat1: A nested list of ints/floats.
        mat2: A nested list of ints/floats.
        axis: The axis along which to concatenate (default 0).

    Returns:
        A new matrix with the matrices concatenated, or None if invalid.
    """
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None
    if axis == 0:
        return [row[:] if isinstance(row, list)
                else row for row in mat1] + \
               [row[:] if isinstance(row, list)
                else row for row in mat2]
    result = [cat_matrices(mat1[i], mat2[i], axis - 1)
              for i in range(len(mat1))]
    if None in result:
        return None
    return result
