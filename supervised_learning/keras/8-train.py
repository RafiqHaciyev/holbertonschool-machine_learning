#!/usr/bin/env python3
"""Module that defines a function to train a model using mini-batch
gradient descent, with optional validation data, early stopping,
learning rate decay, and saving the best model."""
import tensorflow.keras as K


def train_model(
        network, data, labels, batch_size, epochs,
        validation_data=None, early_stopping=False, patience=0,
        learning_rate_decay=False, alpha=0.1, decay_rate=1,
        save_best=False, filepath=None,
        verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent, optionally
    analyzing validation data, applying early stopping, applying
    learning rate decay, and saving the best iteration of the model.

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
        early_stopping (bool): indicates whether early stopping
            should be used. Early stopping should only be performed
            if validation_data exists and is based on validation
            loss.
        patience (int): the patience used for early stopping.
        learning_rate_decay (bool): indicates whether learning rate
            decay should be used. Learning rate decay should only be
            performed if validation_data exists, using inverse time
            decay applied in a stepwise fashion after each epoch.
        alpha (float): the initial learning rate.
        decay_rate (float): the decay rate.
        save_best (bool): indicates whether to save the model after
            each epoch if it is the best. A model is considered the
            best if its validation loss is the lowest that the
            model has obtained.
        filepath (str): the file path where the model should be
            saved.
        verbose (bool): determines if output should be printed
            during training.
        shuffle (bool): determines whether to shuffle the batches
            every epoch.

    Returns:
        History: the History object generated after training the
        model.
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience)
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            """Calculate the learning rate using inverse time decay.

            Args:
                epoch (int): the current epoch number.

            Returns:
                float: the updated learning rate.
            """
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1)
        callbacks.append(lr_decay)

    if save_best and validation_data is not None:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath,
            monitor='val_loss',
            save_best_only=True)
        callbacks.append(checkpoint)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle)

    return history
