#!/usr/bin/env python3
"""Module that builds and evaluates decision trees."""
import numpy as np


class Node:
    """Represents an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node.

        Args:
            feature: the feature used for splitting at this node.
            threshold: the threshold value used for splitting.
            left_child (Node): the left child node.
            right_child (Node): the right child node.
            is_root (bool): whether this node is the root of the tree.
            depth (int): the depth of this node in the tree.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth of the tree below this node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count the nodes below this node.

        Args:
            only_leaves (bool): if True, count only leaf nodes.

        Returns:
            int: the number of nodes (or leaves) below this node.
        """
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves)

        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """Add a prefix to the string representation of the left child.

        Args:
            text (str): the string representation of the left child.

        Returns:
            str: the prefixed string.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Add a prefix to the string representation of the right child.

        Args:
            text (str): the string representation of the right child.

        Returns:
            str: the prefixed string.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """Return a string representation of the node and its children."""
        if self.is_root:
            node_str = "root [feature={}, threshold={}]\n".format(
                self.feature, self.threshold)
        else:
            node_str = "-> node [feature={}, threshold={}]\n".format(
                self.feature, self.threshold)

        left_str = self.left_child_add_prefix(str(self.left_child))
        right_str = self.right_child_add_prefix(str(self.right_child))

        return (node_str + left_str + right_str).rstrip("\n")

    def get_leaves_below(self):
        """Return the list of all leaves below this node.

        Returns:
            list: a list of all Leaf objects below this node.
        """
        return self.left_child.get_leaves_below() + \
            self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """Recursively compute the lower and upper bounds for each
        node in the tree, based on the feature and threshold used
        for splitting.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if child == self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Compute the indicator function for this node from its
        lower and upper bounds, and store it as a lambda in the
        `indicator` attribute.
        """
        def is_large_enough(x):
            """Return a boolean array marking individuals whose
            features are all strictly greater than the lower bounds.

            Args:
                x (numpy.ndarray): array of shape
                    (n_individuals, n_features).

            Returns:
                numpy.ndarray: boolean array of shape (n_individuals,).
            """
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                          for key in list(self.lower.keys())]),
                axis=0)

        def is_small_enough(x):
            """Return a boolean array marking individuals whose
            features are all less than or equal to the upper bounds.

            Args:
                x (numpy.ndarray): array of shape
                    (n_individuals, n_features).

            Returns:
                numpy.ndarray: boolean array of shape (n_individuals,).
            """
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in list(self.upper.keys())]),
                axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Predict the value for a single individual by recursively
        traversing the tree.

        Args:
            x (numpy.ndarray): a 1D array representing a single
                individual's features.

        Returns:
            The predicted value from the appropriate leaf.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Represents a leaf node of a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf.

        Args:
            value: the value predicted by this leaf.
            depth (int): the depth of this leaf in the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count the nodes below this leaf.

        Args:
            only_leaves (bool): if True, count only leaf nodes.

        Returns:
            int: always 1, since a leaf counts itself.
        """
        return 1

    def __str__(self):
        """Return a string representation of the leaf."""
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """Return the list containing this leaf.

        Returns:
            list: a list containing only this Leaf object.
        """
        return [self]

    def update_bounds_below(self):
        """Do nothing, since a leaf has no children to update."""
        pass

    def pred(self, x):
        """Predict the value for a single individual.

        Args:
            x (numpy.ndarray): a 1D array representing a single
                individual's features.

        Returns:
            The value stored in this leaf.
        """
        return self.value


class Decision_Tree():
    """Represents a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree.

        Args:
            max_depth (int): the maximum depth allowed for the tree.
            min_pop (int): the minimum population required to split.
            seed (int): the seed for the random number generator.
            split_criterion (str): the criterion used to split nodes.
            root (Node): the root node of the tree.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the nodes in the decision tree.

        Args:
            only_leaves (bool): if True, count only leaf nodes.

        Returns:
            int: the number of nodes (or leaves) in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the decision tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Return the list of all leaves in the decision tree.

        Returns:
            list: a list of all Leaf objects in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute the lower and upper bounds for every node in the
        decision tree.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """Compute an efficient prediction function for the whole
        decision tree, using the indicator functions of its leaves,
        and store it in the `predict` attribute.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0)

    def pred(self, x):
        """Predict the value for a single individual by recursively
        traversing the tree from the root.

        Args:
            x (numpy.ndarray): a 1D array representing a single
                individual's features.

        Returns:
            The predicted value from the appropriate leaf.
        """
        return self.root.pred(x)

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

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree on the given data.

        Args:
            explanatory (numpy.ndarray): 2D array of shape
                (n_individuals, n_features) containing the features.
            target (numpy.ndarray): 1D array of size n_individuals
                containing the target class for each individual.
            verbose (int): if 1, print training statistics.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {
                self.accuracy(self.explanatory, self.target)}""")

    def fit_node(self, node):
        """Recursively split a node until stopping criteria are met.

        Args:
            node (Node): the node to split.
        """
        node.feature, node.threshold = self.split_criterion(node)

        above_threshold = self.explanatory[:, node.feature] > \
            node.threshold
        left_population = node.sub_population & above_threshold
        right_population = node.sub_population & ~above_threshold

        is_left_leaf = (node.depth + 1 == self.max_depth or
                        np.sum(left_population) < self.min_pop or
                        np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth + 1 == self.max_depth or
                         np.sum(right_population) < self.min_pop or
                         np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child for a node.

        Args:
            node (Node): the parent node.
            sub_population (numpy.ndarray): boolean array marking
                which individuals belong to this leaf.

        Returns:
            Leaf: the newly created leaf.
        """
        values, counts = np.unique(
            self.target[sub_population], return_counts=True)
        value = values[np.argmax(counts)]
        leaf_child = Leaf(value)
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

    def accuracy(self, test_explanatory, test_target):
        """Compute the accuracy of the tree's predictions.

        Args:
            test_explanatory (numpy.ndarray): 2D array of features.
            test_target (numpy.ndarray): 1D array of true targets.

        Returns:
            float: the fraction of correct predictions.
        """
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size

    def possible_thresholds(self, node, feature):
        """Compute the possible thresholds for splitting on a feature.

        Args:
            node (Node): the node to compute thresholds for.
            feature (int): the feature index.

        Returns:
            numpy.ndarray: the midpoints between consecutive unique
                feature values observed in the node's sub-population.
        """
        values = np.unique(
            (self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Compute the best threshold and average Gini impurity for
        splitting a node on a single feature.

        Args:
            node (Node): the node to compute a split for.
            feature (int): the feature index to split on.

        Returns:
            tuple: (best_threshold, best_gini_average) for this
                feature.
        """
        thresholds = self.possible_thresholds(node, feature)
        x = self.explanatory[:, feature][node.sub_population]
        target_sub = self.target[node.sub_population]
        classes = np.unique(target_sub)

        left_F = (x[:, None, None] > thresholds[None, :, None]) & \
            (target_sub[:, None, None] == classes[None, None, :])
        right_F = (x[:, None, None] <= thresholds[None, :, None]) & \
            (target_sub[:, None, None] == classes[None, None, :])

        left_counts = np.sum(left_F, axis=0)
        right_counts = np.sum(right_F, axis=0)

        left_totals = np.sum(left_counts, axis=1)
        right_totals = np.sum(right_counts, axis=1)

        left_gini = 1 - np.sum(
            (left_counts / left_totals[:, None]) ** 2, axis=1)
        right_gini = 1 - np.sum(
            (right_counts / right_totals[:, None]) ** 2, axis=1)

        n = target_sub.size
        gini_avg = (left_totals / n) * left_gini + \
            (right_totals / n) * right_gini

        best_idx = np.argmin(gini_avg)
        return thresholds[best_idx], gini_avg[best_idx]

    def Gini_split_criterion(self, node):
        """Choose the best feature and threshold to split a node,
        based on the average Gini impurity across all features.

        Args:
            node (Node): the node to compute a split for.

        Returns:
            tuple: (feature, threshold) that minimizes the average
                Gini impurity.
        """
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
