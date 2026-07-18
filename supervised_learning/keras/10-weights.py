#!/usr/bin/env python3
"""Module that defines functions to save and load a model's
weights."""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save a model's weights.

    Args:
        network (keras.Model): the model whose weights should be
            saved.
        filename (str): the path of the file that the weights
            should be saved to.
        save_format (str): the format in which the weights should
            be saved.

    Returns:
        None.
    """
    network.save_weights(filename, save_format=save_format)

    return None


def load_weights(network, filename):
    """Load a model's weights.

    Args:
        network (keras.Model): the model to which the weights
            should be loaded.
        filename (str): the path of the file that the weights
            should be loaded from.

    Returns:
        None.
    """
    network.load_weights(filename)

    return None
