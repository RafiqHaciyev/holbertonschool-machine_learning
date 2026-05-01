#!/usr/bin/env python3
"""Module for calculating the determinant of a matrix."""


def determinant(matrix):
    """Calculate the determinant of a matrix.

    Args:
        matrix: a list of lists whose determinant should be calculated.

    Returns:
        The determinant of the matrix.

    Raises:
        TypeError: if matrix is not a list of lists.
        ValueError: if matrix is not square.
    """
    if not isinstance(matrix, list) or len(matrix) == 0 or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub = [
            [matrix[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(sub)

    return det
