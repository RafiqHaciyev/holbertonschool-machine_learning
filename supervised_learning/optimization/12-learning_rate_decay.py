#!/usr/bin/env python3
"""Module that creates an inverse time decay operation in TensorFlow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Creates a learning rate decay operation using inverse time decay

    Args:
        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha decays
        decay_step: number of passes of gradient descent that should occur
            before alpha is decayed further

    Returns:
        the learning rate decay operation
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
