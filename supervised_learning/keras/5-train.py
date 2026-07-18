#!/usr/bin/env python3
"""Module that defines a function to train a model using mini-batch
gradient descent, with optional validation data."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                 validation_data=None, verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent, optionally
    analyzing validation data.

    Args:
        network (keras.Model): the model to train.
        data (numpy.ndarray): array of shape (m, nx) containing the
            input data.
        labels (numpy.ndarray): one-hot array of shape (m, classes)
            containing the labels of data.
        batch_size (int): the size of the batch used for mini-batch
            gradient descent.
        epochs (int): the number of passes through data for
            mini-batch gradient descent.
        validation_data: the data to validate the model with, if
            not None.
        verbose (bool): determines if output should be printed
            during training.
        shuffle (bool): determines whether to shuffle the batches
            every epoch.

    Returns:
        History: the History object generated after training the
        model.
    """
    history = network.fit(x=data,
                           y=labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=validation_data,
                           verbose=verbose,
                           shuffle=shuffle)

    return history
