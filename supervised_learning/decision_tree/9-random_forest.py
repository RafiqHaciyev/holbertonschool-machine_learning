#!/usr/bin/env python3
"""Module that implements a random forest classifier built from
multiple randomly-split decision trees.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Represents a random forest of decision trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize a Random_Forest.

        Args:
            n_trees (int): the number of trees to build in the forest.
            max_depth (int): the maximum depth allowed for each tree.
            min_pop (int): the minimum population required to split.
            seed (int): the seed for the random number generator.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict the class of each individual by majority vote
        across all trees in the forest.

        Args:
            explanatory (numpy.ndarray): 2D array of shape
                (n_individuals, n_features) containing the features.

        Returns:
            numpy.ndarray: 1D array of size n_individuals containing
                the most frequent prediction across all trees for
                each individual.
        """
        predictions = np.array(
            [f(explanatory) for f in self.numpy_preds])

        def mode(x):
            """Return the most frequent value in a 1D array."""
            return np.bincount(x.astype(int)).argmax()

        return np.apply_along_axis(mode, axis=0, arr=predictions)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Train the random forest on the given data.

        Args:
            explanatory (numpy.ndarray): 2D array of shape
                (n_individuals, n_features) containing the features.
            target (numpy.ndarray): 1D array of size n_individuals
                containing the target class for each individual.
            n_trees (int): the number of trees to build.
            verbose (int): if 1, print training statistics.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth, min_pop=self.min_pop,
                seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {
                self.accuracy(self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """Compute the accuracy of the forest's predictions.

        Args:
            test_explanatory (numpy.ndarray): 2D array of features.
            test_target (numpy.ndarray): 1D array of true targets.

        Returns:
            float: the fraction of correct predictions.
        """
        return np.sum(np.equal(
            self.predict(test_explanatory),
            test_target)) / test_target.size
