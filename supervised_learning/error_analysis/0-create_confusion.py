#!/usr/bin/env python3
"""Module that creates a confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create a confusion matrix from one-hot labels and predictions.

    Args:
        labels (numpy.ndarray): one-hot array of shape (m, classes)
            containing the correct labels for each data point.
        logits (numpy.ndarray): one-hot array of shape (m, classes)
            containing the predicted labels.

    Returns:
        numpy.ndarray: confusion matrix of shape (classes, classes)
            with row indices representing the correct labels and
            column indices representing the predicted labels.
    """
    return np.matmul(labels.T, logits)
