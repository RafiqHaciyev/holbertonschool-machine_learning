#!/usr/bin/env python3
"""Module that sets up gradient descent with momentum in TensorFlow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Sets up the gradient descent with momentum optimization algorithm

    Args:
        alpha: learning rate
        beta1: momentum weight

    Returns:
        optimizer
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
