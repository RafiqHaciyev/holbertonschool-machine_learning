#!/usr/bin/env python3
"""Module that calculates the specificity for each class in a
confusion matrix.
"""
import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix of shape
            (classes, classes) where row indices represent the
            correct labels and column indices represent the
            predicted labels.

    Returns:
        numpy.ndarray: array of shape (classes,) containing the
            specificity of each class.
    """
    true_positives = np.diagonal(confusion)
    actual = np.sum(confusion, axis=1)
    predicted = np.sum(confusion, axis=0)
    total = np.sum(confusion)

    false_positives = predicted - true_positives
    false_negatives = actual - true_positives
    true_negatives = total - true_positives - false_positives - \
        false_negatives

    return true_negatives / (true_negatives + false_positives)
