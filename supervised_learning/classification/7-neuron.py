#!/usr/bin/env python3
"""Defines a single neuron performing binary classification."""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron.

        Args:
            nx (int): the number of input features to the neuron.

        Raises:
            TypeError: if nx is not an integer.
            ValueError: if nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Get the weights vector of the neuron."""
        return self.__W

    @property
    def b(self):
        """Get the bias of the neuron."""
        return self.__b

    @property
    def A(self):
        """Get the activated output of the neuron."""
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): shape (nx, m) that contains the input data.

        Returns:
            The private attribute __A.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): shape (1, m) that contains the correct
                labels for the input data.
            A (numpy.ndarray): shape (1, m) containing the activated
                output of the neuron for each example.

        Returns:
            The cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions.

        Args:
            X (numpy.ndarray): shape (nx, m) that contains the input data.
            Y (numpy.ndarray): shape (1, m) that contains the correct
                labels for the input data.

        Returns:
            The neuron's prediction and the cost of the network.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): shape (nx, m) that contains the input data.
            Y (numpy.ndarray): shape (1, m) that contains the correct
                labels for the input data.
            A (numpy.ndarray): shape (1, m) containing the activated
                output of the neuron for each example.
            alpha (float): the learning rate.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neuron.

        Args:
            X (numpy.ndarray): shape (nx, m) that contains the input data.
            Y (numpy.ndarray): shape (1, m) that contains the correct
                labels for the input data.
            iterations (int): the number of iterations to train over.
            alpha (float): the learning rate.
            verbose (bool): defines whether or not to print information
                about the training.
            graph (bool): defines whether or not to graph information
                about the training once the training has completed.
            step (int): the step interval used for printing/graphing.

        Raises:
            TypeError: if iterations is not an integer.
            ValueError: if iterations is not positive.
            TypeError: if alpha is not a float.
            ValueError: if alpha is not positive.
            TypeError: if step is not an integer.
            ValueError: if step is not positive or is greater than
                iterations.

        Returns:
            The evaluation of the training data after iterations of
            training have occurred.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    steps.append(i)

            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
