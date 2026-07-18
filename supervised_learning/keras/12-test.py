#!/usr/bin/env python3
"""Module that defines a function to test a neural network."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test a neural network.

    Args:
        network (keras.Model): the network model to test.
        data (numpy.ndarray): the input data to test the model
            with.
        labels (numpy.ndarray): the correct one-hot labels of
            data.
        verbose (bool): determines if output should be printed
            during the testing process.

    Returns:
        list: the loss and accuracy of the model with the testing
        data, respectively.
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
