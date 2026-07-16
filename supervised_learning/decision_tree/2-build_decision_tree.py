#!/usr/bin/env python3
"""Build a Decision Tree from Node, Leaf and Decision_Tree classes."""
import numpy as np


class Node:
    """Represents an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def __str__(self):
        """Return a human-readable string representation of the tree."""
        if self.is_root:
            s = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            s = f"-> node [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            s += self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            s += self.right_child_add_prefix(self.right_child.__str__())
        return s

    def left_child_add_prefix(self, text):
        """Add the left-branch prefix ('|') to each line of text."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Add the right-branch prefix (blank) to each line of text."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)


class Leaf(Node):
    """Represents a leaf node of a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Return a human-readable string representation of the leaf."""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """Represents a full decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree."""
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

    def __str__(self):
        """Return a human-readable string representation of the tree."""
        return self.root.__str__()
