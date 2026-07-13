#!/usr/bin/env python3
"""Module that implements an isolation random tree, used to detect
outliers in a dataset with no target labels.
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Represents an isolation random tree."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize an Isolation_Random_Tree.

        Args:
            max_depth (int): the maximum depth allowed for the tree.
            seed (int): the seed for the random number generator.
            root (Node): the root node of the tree.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Return a string representation of the isolation tree."""
        return self.root.__str__()

    def depth(self):
        """Return the maximum depth of the isolation tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the nodes in the isolation tree.

        Args:
            only_leaves (bool): if True, count only leaf nodes.

        Returns:
            int: the number of nodes (or leaves) in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Compute the lower and upper bounds for every node in the
        isolation tree.
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return the list of all leaves in the isolation tree.

        Returns:
            list: a list of all Leaf objects in the tree.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """Compute an efficient prediction function for the whole
        isolation tree, using the indicator functions of its leaves,
        and store it in the `predict` attribute.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0)

    def np_extrema(self, arr):
        """Return the minimum and maximum values of an array.

        Args:
            arr (numpy.ndarray): the array to compute extrema on.

        Returns:
            tuple: (min, max) of the array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly choose a feature and threshold to split a node.

        Args:
            node (Node): the node to compute a split for.

        Returns:
            tuple: (feature, threshold) chosen at random.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child for a node. The leaf's value is the
        depth at which it was created, since isolation trees have
        no target classes to predict.

        Args:
            node (Node): the parent node.
            sub_population (numpy.ndarray): boolean array marking
                which individuals belong to this leaf.

        Returns:
            Leaf: the newly created leaf.
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create a node child for a node.

        Args:
            node (Node): the parent node.
            sub_population (numpy.ndarray): boolean array marking
                which individuals belong to this node.

        Returns:
            Node: the newly created node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively split a node until stopping criteria are met.

        A node becomes a leaf if either the maximum depth is reached
        or its population has shrunk to the minimum population size,
        since there is no target-based homogeneity check available.

        Args:
            node (Node): the node to split.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        above_threshold = self.explanatory[:, node.feature] > \
            node.threshold
        left_population = node.sub_population & above_threshold
        right_population = node.sub_population & ~above_threshold

        is_left_leaf = (node.depth + 1 == self.max_depth or
                        np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth + 1 == self.max_depth or
                         np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Train the isolation tree on the given data.

        Args:
            explanatory (numpy.ndarray): 2D array of shape
                (n_individuals, n_features) containing the features.
            verbose (int): if 1, print training statistics.
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(
            explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
