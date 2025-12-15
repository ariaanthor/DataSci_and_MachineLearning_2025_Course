"""
decision_tree.py

A lightweight implementation of a CART-style decision tree classifier.

The classifier supports Gini impurity, depth limits, minimum sample
constraints, and optional feature subsampling.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class _TreeNode:
    """
    Node container used internally by the decision tree.

    A node may either represent:
    - a decision point (feature + threshold with left/right children), or
    - a terminal leaf (class probability vector only).

    Attributes
    ----------
    feature_index : int or None
        Index of the feature used for splitting at this node.
        If None, the node is treated as a leaf.
    threshold : float or None
        Split value: samples with feature <= threshold go left.
    left : _TreeNode or None
        Child node for samples satisfying the split condition.
    right : _TreeNode or None
        Child node for samples not satisfying the split condition.
    proba : np.ndarray or None
        Class probability distribution stored at this node.
        Used directly when the node is a leaf.
    """
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    proba: Optional[np.ndarray] = None

    def is_leaf(self) -> bool:
        """Return True if this node represents a terminal leaf."""
        return self.feature_index is None


class DecisionTreeClassifier:
    """
    Classification decision tree based on the CART framework.

    The tree is grown recursively using Gini impurity to evaluate candidate
    splits. The interface mirrors common ML libraries but the internals
    are deliberately straightforward for instructional use.

    Parameters
    ----------
    max_depth : int or None
        Maximum allowed depth of the tree. If None, the tree grows until
        other stopping criteria are met.
    min_samples_split : int
        Minimum number of samples required to attempt a split.
    min_samples_leaf : int
        Minimum number of samples that must remain in each child node.
    max_features : int, float, or None
        Controls how many features are considered at each split:
        - int: exact number of features
        - float: fraction of total features
        - None: use all features
    random_state : int or None
        Seed for feature subsampling randomness.

    Attributes
    ----------
    n_classes_ : int
        Number of distinct target classes.
    n_features_ : int
        Number of input features.
    tree_ : _TreeNode
        Root of the trained decision tree.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[float | int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        # Hyperparameters controlling tree growth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        # Attributes populated during fitting
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.tree_: Optional[_TreeNode] = None
        self._rng: Optional[np.random.Generator] = None

    # ==============================================================
    # Public interface
    # ==============================================================
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Train the decision tree on labeled data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.
        y : np.ndarray, shape (n_samples,)
            Integer-encoded class labels (0, 1, ..., K-1).

        Returns
        -------
        self
            The fitted classifier.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Basic shape validation
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of class labels.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        # Record dataset characteristics
        self.n_features_ = X.shape[1]

        # Enforce integer-coded class labels for simplicity
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer-encoded class labels (0, 1, 2, ...).")

        self.n_classes_ = int(np.max(y) + 1)

        # Initialize RNG for feature subsampling
        self._rng = np.random.default_rng(self.random_state)

        # Recursively construct the tree
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class indices.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probability distributions for input samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes_)
            Per-sample class probabilities.
        """
        if self.tree_ is None or self.n_classes_ is None:
            raise RuntimeError("The classifier has not been fitted yet.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.n_classes_), dtype=float)

        # Route each sample through the tree independently
        for i in range(n_samples):
            leaf = self._traverse_tree(X[i], self.tree_)
            out[i] = leaf.proba

        return out

    # ==============================================================
    # Tree construction helpers
    # ==============================================================
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        """
        Recursively expand the tree from the current node.
        """
        n_samples, _ = X.shape
        unique_labels = np.unique(y)

        # Compute label distribution at this node
        node_proba = self._class_proba(y)

        # Stop if pure, too deep, or insufficient samples
        if (
            unique_labels.size == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
        ):
            return _TreeNode(proba=node_proba)

        # Search for the optimal split
        feat, thresh, (left_mask, right_mask) = self._best_split(X, y)

        # If no valid split is found, create a leaf
        if feat is None:
            return _TreeNode(proba=node_proba)

        # Recursively build subtrees
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            feature_index=feat,
            threshold=thresh,
            left=left,
            right=right,
            proba=node_proba,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], Tuple[np.ndarray, np.ndarray]]:
        """
        Identify the split (feature + threshold) that minimizes Gini impurity.
        """
        n_samples, n_features = X.shape

        # Ensure leaves can satisfy minimum size constraints
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, (np.array([]), np.array([]))

        # Select candidate features
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif isinstance(self.max_features, int):
            if self.max_features <= 0 or self.max_features > n_features:
                raise ValueError("max_features int must be in [1, n_features].")
            feature_indices = self._rng.choice(n_features, self.max_features, replace=False)
        elif isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):
                raise ValueError("max_features float must be in (0, 1].")
            k = max(1, int(self.max_features * n_features))
            feature_indices = self._rng.choice(n_features, k, replace=False)
        else:
            raise ValueError("max_features must be None, int, or float.")

        best_score = 1.0
        best_feat = None
        best_thresh = None
        best_left = np.array([], dtype=bool)
        best_right = np.array([], dtype=bool)

        # Evaluate all candidate splits
        for feat in feature_indices:
            column = X[:, feat]
            for thresh in np.unique(column):
                left_mask = column <= thresh
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                g_left = self._gini(y[left_mask])
                g_right = self._gini(y[right_mask])
                g_total = (n_left * g_left + n_right * g_right) / n_samples

                if g_total < best_score:
                    best_score = g_total
                    best_feat = feat
                    best_thresh = float(thresh)
                    best_left = left_mask
                    best_right = right_mask

        if best_feat is None:
            return None, None, (np.array([]), np.array([]))

        return best_feat, best_thresh, (best_left, best_right)

    # ==============================================================
    # Utility routines
    # ==============================================================

    def _gini(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity for a set of labels.
        """
        if y.size == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def _class_proba(self, y: np.ndarray) -> np.ndarray:
        """
        Convert a label vector into a normalized class-frequency vector.
        """
        counts = np.bincount(y, minlength=self.n_classes_)
        total = counts.sum()
        if total == 0:
            # Fallback safeguard (should not normally occur)
            return np.full(self.n_classes_, 1.0 / self.n_classes_)
        return counts / total

    def _traverse_tree(self, x: np.ndarray, node: _TreeNode) -> _TreeNode:
        """
        Follow decision rules from the root to a leaf for a single sample.
        """
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node
