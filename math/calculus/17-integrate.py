#!/usr/bin/env python3
"""
Module for polynomial integration.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.
    Args:
        poly: list of coefficients where index is the power of x.
        C: integration constant.
    Returns:
        A new list of coefficients representing the integral.
    """
    # 1. Validate that poly is a list AND is not empty
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # 2. Validate that every item in the list is a number
    if not all(isinstance(n, (int, float)) for n in poly):
        return None

    # 3. Validate that C is a number
    if not isinstance(C, (int, float)):
        return None

    # Initialize with the constant C
    integral = [C]

    # Perform integration
    for i in range(len(poly)):
        new_val = poly[i] / (i + 1)

        # Convert to integer if it's a whole number
        if new_val == int(new_val):
            new_val = int(new_val)

        integral.append(new_val)

    # Ensure the list is as small as possible by removing trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
