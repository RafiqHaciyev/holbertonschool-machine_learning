#!/usr/bin/env python3
"""Module that implements an isolation random forest, used to detect
outliers in a dataset with no target labels.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Represents a forest of isolation random trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize an Isolation_Random_Forest.

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
        self.seed = seed

    def predict(self, explanatory):
        """Predict the mean isolation depth of each individual across
        all trees in the forest.

        Args:
            explanatory (numpy.ndarray): 2D array of shape
                (n_individuals, n_features) containing the features.

        Returns:
            numpy.ndarray: 1D array of size n_individuals containing
                the mean leaf depth across all trees for each
                individual.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Train the isolation random forest on the given data.

        Args:
            explanatory (numpy.ndarray): 2D array of shape
                (n_individuals, n_features) containing the features.
            n_trees (int): the number of trees to build.
            verbose (int): if 1, print training statistics.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth, seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """Return the n_suspects rows in explanatory that have the
        smallest mean depth (the individuals most likely to be
        outliers).

        Args:
            explanatory (numpy.ndarray): 2D array of shape
                (n_individuals, n_features) containing the features.
            n_suspects (int): the number of suspects to return.

        Returns:
            tuple: (suspects, depths) where suspects is a 2D array
                of the n_suspects rows with the smallest mean depth,
                and depths is a 1D array of their corresponding mean
                depths, sorted in ascending order.
        """
        depths = self.predict(explanatory)
        order = np.argsort(depths)
        suspect_indices = order[:n_suspects]
        return explanatory[suspect_indices], depths[suspect_indices]
