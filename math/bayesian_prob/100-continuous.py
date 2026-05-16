#!/usr/bin/env python3
"""Module for calculating the continuous posterior probability."""
from scipy import special


def posterior(x, n, p1, p2):
    """Calculate the posterior probability that p is within [p1, p2].

    Assumes a uniform prior on p, making the posterior a Beta distribution
    with parameters alpha = x + 1 and beta = n - x + 1.

    Args:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        p1 (float): Lower bound on the range.
        p2 (float): Upper bound on the range.

    Returns:
        float: Posterior probability that p is within [p1, p2].

    Raises:
        ValueError: If n is not a positive integer.
        ValueError: If x is not an integer >= 0.
        ValueError: If x > n.
        ValueError: If p1 is not a float in the range [0, 1].
        ValueError: If p2 is not a float in the range [0, 1].
        ValueError: If p2 <= p1.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # With uniform prior, posterior is Beta(x + 1, n - x + 1)
    alpha = x + 1
    beta = n - x + 1

    # P(p1 <= p <= p2) = CDF(p2) - CDF(p1) using regularized incomplete beta
    cdf_p2 = special.btdtr(alpha, beta, p2)
    cdf_p1 = special.btdtr(alpha, beta, p1)

    return cdf_p2 - cdf_p1
