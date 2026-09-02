#!/usr/bin/env python3
"""Module that updates a variable using gradient descent with
momentum optimization.
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using the gradient descent with momentum
    optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        v: The previous first moment of var.

    Returns:
        The updated variable and the new moment, respectively.
    """
    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new
    return var_new, v_new
