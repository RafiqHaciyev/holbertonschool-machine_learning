#!/usr/bin/env python3
"""
Module for polynomial integration.
"""


def poly_integral(poly, C=0):  # Change 'c' to 'C' here
    """
    Calculates the integral of a polynomial.
    Args:
        poly: list of coefficients where index is the power of x.
        C: integration constant.
    Returns:
        A new list of coefficients representing the integral.
    """
    # Use capital C in the validation and initialization as well
    if not isinstance(poly, list) or not all(isinstance(n, (int, float))
                                             for n in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    if not poly:
        return [C]

    integral = [C]

    for i in range(len(poly)):
        new_val = poly[i] / (i + 1)

        if new_val == int(new_val):
            new_val = int(new_val)

        integral.append(new_val)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
