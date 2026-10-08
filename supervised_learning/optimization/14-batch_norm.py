#!/usr/bin/env python3
"""Module that creates a batch normalization layer in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function to use on the output of the layer

    Returns:
        a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)
    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )
    Z_norm = batch_norm(Z, training=True)
    return activation(Z_norm)
