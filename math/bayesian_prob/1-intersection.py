#!/usr/bin/env python3
"""Module for calculating the intersection of data and hypothetical probs."""
import numpy as np


def likelihood(x, n, P):
    """Calculate likelihood of obtaining data given hypothetical probabilities.

    Args:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array of hypothetical probabilities of
            developing severe side effects.

    Returns:
        numpy.ndarray: 1D array of likelihoods for each probability in P.
    """
    coeff = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )
    return coeff * (P ** x) * ((1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """Calculate the intersection of obtaining data with hypothetical probs.

    Args:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array of hypothetical probabilities of
            developing severe side effects.
        Pr (numpy.ndarray): 1D array of prior beliefs of P.

    Returns:
        numpy.ndarray: 1D array of intersections for each probability in P.

    Raises:
        ValueError: If n is not a positive integer.
        ValueError: If x is not an integer >= 0.
        ValueError: If x > n.
        TypeError: If P is not a 1D numpy.ndarray.
        TypeError: If Pr is not a numpy.ndarray with the same shape as P.
        ValueError: If any value in P or Pr is not in the range [0, 1].
        ValueError: If Pr does not sum to 1.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    return likelihood(x, n, P) * Pr
