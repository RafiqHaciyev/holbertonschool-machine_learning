#!/usr/bin/env python3
"""Module for calculating the shape of a matrix."""


def matrix_shape(matrix):
    """Calculate the shape of a matrix.

    Args:
        matrix: A nested list representing a matrix.

    Returns:
        A list of integers representing the shape of the matrix.
    """
    shape = []
    current = matrix
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0]
    return shape
