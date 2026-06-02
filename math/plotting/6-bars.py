#!/usr/bin/env python3
"""Module for plotting a stacked bar graph of fruit per person."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar graph of fruit quantities per person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    bottom = np.zeros(3)

    for i, (color, label) in enumerate(zip(colors, labels)):
        plt.bar(people, fruit[i], width=0.5, bottom=bottom,
                color=color, label=label)
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.yticks(range(0, 81, 10))
    plt.ylim(0, 80)
    plt.legend()
    plt.show()
