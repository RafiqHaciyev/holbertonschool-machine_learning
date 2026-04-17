#!/usr/bin/env python3
"""
Module for polynomial integration.
"""


def poly_integral(poly, c=0):
    """
    Calculates the integral of a polynomial.
    Args:
        poly: list of coefficients where index is the power of x.
        c: integration constant.
    Returns:
        A new list of coefficients representing the integral.
    """
    # Validation
    if not isinstance(poly, list) or not all(isinstance(n, (int, float))
                                             for n in poly):
        return None
    if not isinstance(c, (int, float)):
        return None

    # Edge case: if poly is empty, return just the constant
    if not poly:
        return [c]

    # Initialize the integral list with the constant C at index 0
    # Every original power x^i becomes x^(i+1)
    integral = [c]

    for i in range(len(poly)):
        # Calculate new coefficient: coeff / (power + 1)
        new_val = poly[i] / (i + 1)

        # Convert to int if it's a whole number (e.g., 5.0 -> 5)
        if new_val == int(new_val):
            new_val = int(new_val)

        integral.append(new_val)

    # Requirements: The returned list should be as small as possible.
    # We remove trailing zeros, but ensure at least index 0 remains.
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
