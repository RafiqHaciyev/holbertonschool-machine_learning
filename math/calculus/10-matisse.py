#!/usr/bin/env python3
"""
This module contains a function to calculate the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.
    Args:
        poly: A list of coefficients where index = power.
    Returns:
        A new list of coefficients representing the derivative.
    """
    # Validation: poly must be a non-empty list of numbers
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(n, (int, float)) for n in poly):
        return None

    # If the polynomial is a constant (e.g., [5]), the derivative is [0]
    if len(poly) == 1:
        return [0]

    derivative = []

    # Start from index 1 because the derivative of index 0 (constant) is 0
    for i in range(1, len(poly)):
        new_coeff = poly[i] * i
        derivative.append(new_coeff)

    # Requirements check: if the list is empty or all zeros, return [0]
    if not derivative or all(c == 0 for c in derivative):
        return [0]

    return derivative
