#!/usr/bin/env python3
"""Module that calculates the weighted moving average of a data set."""


def moving_average(data, beta):
    """Calculate the weighted moving average of a data set.

    Args:
        data (list): The list of data to calculate the moving average
            of.
        beta (float): The weight used for the moving average.

    Returns:
        list: The moving averages of data, using bias correction.
    """
    moving_averages = []
    v = 0
    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        bias_corrected = v / (1 - beta ** (i + 1))
        moving_averages.append(bias_corrected)
    return moving_averages
