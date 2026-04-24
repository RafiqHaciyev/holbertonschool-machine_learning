#!/usr/bin/env python3
"""Module for transposing a 2D matrix."""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix.

    Args:
        matrix: A 2D list representing a matrix.

    Returns:
        A new 2D list representing the transposed matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
