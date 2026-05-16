#!/usr/bin/env python3
"""Module for the Multivariate Normal distribution class."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Initialize a Multivariate Normal distribution.

        Args:
            data (numpy.ndarray): Array of shape (d, n) containing the
                data set, where d is the number of dimensions and n is
                the number of data points.

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If n is less than 2.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.sum(data, axis=1, keepdims=True) / n
        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)
