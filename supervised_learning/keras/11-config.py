#!/usr/bin/env python3
"""Module that defines functions to save and load a model's
configuration in JSON format."""
import tensorflow.keras as K


def save_config(network, filename):
    """Save a model's configuration in JSON format.

    Args:
        network (keras.Model): the model whose configuration
            should be saved.
        filename (str): the path of the file that the
            configuration should be saved to.

    Returns:
        None.
    """
    json_config = network.to_json()

    with open(filename, 'w') as f:
        f.write(json_config)

    return None


def load_config(filename):
    """Load a model with a specific configuration.

    Args:
        filename (str): the path of the file containing the
            model's configuration in JSON format.

    Returns:
        keras.Model: the loaded model.
    """
    with open(filename, 'r') as f:
        json_config = f.read()

    return K.models.model_from_json(json_config)
