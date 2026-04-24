#!/usr/bin/env python3
"""Module for slicing a numpy.ndarray along specific axes."""


def np_slice(matrix, axes={}):
    """Slice a matrix along specific axes.

    Args:
        matrix: A numpy.ndarray.
        axes: A dictionary where keys are axes and values are slice tuples.

    Returns:
        A new numpy.ndarray sliced along the specified axes.
    """
    slices = [slice(None)] * matrix.ndim
    for axis, s in axes.items():
        slices[axis] = slice(*s)
    return matrix[tuple(slices)]
