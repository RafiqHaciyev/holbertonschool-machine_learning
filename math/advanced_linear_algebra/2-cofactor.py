#!/usr/bin/env python3
"""Module for calculating the cofactor matrix of a matrix."""


def cofactor(matrix):
    """Calculate the cofactor matrix of a matrix.

    Args:
        matrix: a list of lists whose cofactor matrix should be calculated.

    Returns:
        The cofactor matrix of the matrix.

    Raises:
        TypeError: if matrix is not a list of lists.
        ValueError: if matrix is not square or is empty.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    result = []
    for i in range(n):
        row = []
        for j in range(n):
            sub = [
                [matrix[r][c] for c in range(n) if c != j]
                for r in range(n) if r != i
            ]
            sign = (-1) ** (i + j)
            row.append(sign * _det(sub))
        result.append(row)

    return result


def _det(matrix):
    """Calculate the determinant of a matrix (internal helper).

    Args:
        matrix: a list of lists.

    Returns:
        The determinant of the matrix.
    """
    n = len(matrix)
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
        det += ((-1) ** j) * matrix[0][j] * _det(sub)
    return det
