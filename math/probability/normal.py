#!/usr/bin/env python3
"""Module for Normal distribution"""


class Normal:
    """Represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes the Normal distribution

        Args:
            data (list): list of data to estimate the distribution
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution

        Raises:
            TypeError: if data is not a list
            ValueError: if data does not contain at least two data points
            ValueError: if stddev is not a positive value
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)
