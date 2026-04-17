#!/usr/bin/env python3
"""
This module provides a function to calculate the sum of squares.
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n using the mathematical formula.
    Args:
        n: The stopping condition.
    Returns:
        The integer value of the sum, or None if n is invalid.
    """
    if not isinstance(n, (int, float)) or n < 1:
        return None

    # Using the formula: (n * (n + 1) * (2n + 1)) / 6
    # We use integer division // to ensure the return value is an int
    result = (n * (n + 1) * (2 * n + 1)) // 6
    return int(result)
