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
