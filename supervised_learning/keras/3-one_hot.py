#!/usr/bin/env python3
"""Module that defines a function to convert a label vector into a
one-hot matrix."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Convert a label vector into a one-hot matrix.

    Args:
        labels: the label vector to convert.
        classes (int): the number of classes. If None, the number of
            classes is inferred from the labels.

    Returns:
        The one-hot matrix, where the last dimension is the number
        of classes.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
