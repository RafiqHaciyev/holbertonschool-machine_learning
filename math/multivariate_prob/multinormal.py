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

    def pdf(self, x):
        """Calculate the PDF at a data point.

        Args:
            x (numpy.ndarray): Array of shape (d, 1) containing the data
                point whose PDF should be calculated.

        Returns:
            float: The value of the PDF at x.

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x does not have shape (d, 1).
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        diff = x - self.mean
        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)

        norm = 1 / np.sqrt(((2 * np.pi) ** d) * cov_det)
        exponent = -0.5 * (diff.T @ cov_inv @ diff)[0, 0]

        return float(norm * np.exp(exponent))
