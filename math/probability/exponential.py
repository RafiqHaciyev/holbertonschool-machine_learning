#!/usr/bin/env python3
"""Module for Exponential distribution"""


class Exponential:
    """Represents an Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initializes the Exponential distribution

        Args:
            data (list): list of data to estimate the distribution
            lambtha (float): expected number of occurrences in a time frame

        Raises:
            TypeError: if data is not a list
            ValueError: if data does not contain at least two data points
            ValueError: if lambtha is not a positive value
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))
