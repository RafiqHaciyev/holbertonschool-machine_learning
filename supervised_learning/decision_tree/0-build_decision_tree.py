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
