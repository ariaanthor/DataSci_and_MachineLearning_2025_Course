"""
Decision Tree Regression.

This module implements a basic regression tree using a greedy,
top-down CART-style algorithm. Splits are chosen to minimize
mean squared error (variance) in the target values.

The goal of this implementation is clarity and instructional value
rather than raw performance.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Sequence

import numpy as np

__all__ = ["DecisionTreeRegressor"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ---------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert input to a non-empty 2D NumPy array of floats.
    """
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; received {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{name} must contain numeric values.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


def _ensure_1d_float(y: ArrayLike, name: str = "y") -> np.ndarray:
    """
    Convert input to a non-empty 1D NumPy array of floats.
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; received {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{name} must contain numeric values.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


# ---------------------------------------------------------------------
# Tree node definition
# ---------------------------------------------------------------------

@dataclass
class _RegressionNode:
    """
    Internal representation of a regression tree node.

    A node is either:
    - a decision node (feature_index + threshold defined), or
    - a leaf node (stores a scalar prediction).
    """
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_RegressionNode"] = None
    right: Optional["_RegressionNode"] = None
    value: Optional[float] = None

    def is_leaf(self) -> bool:
        """Return True if this node is terminal."""
        return self.feature_index is None


# ---------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------

class DecisionTreeRegressor:
    """
    Regression tree using variance reduction (MSE).

    The tree recursively partitions the feature space into axis-aligned
    regions and predicts the mean target value within each region.

    Parameters
    ----------
    max_depth : int or None
        Maximum allowed depth of the tree.
    min_samples_split : int
        Minimum number of samples required to attempt a split.
    min_samples_leaf : int
        Minimum samples required in each child after a split.
    max_features : int, float, or None
        Number (or fraction) of features to consider at each split.
    random_state : int or None
        Random seed for feature subsampling.

    Notes
    -----
    - Leaves store the mean of the training targets.
    - Splits are only accepted if they reduce variance.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.n_features_: Optional[int] = None
        self.tree_: Optional[_RegressionNode] = None
        self._rng: Optional[np.random.Generator] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DecisionTreeRegressor":
        """
        Train the regression tree.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Feature matrix.
        y : array_like of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_float(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        self.n_features_ = X_arr.shape[1]
        self._rng = np.random.default_rng(self.random_state)

        self.tree_ = self._grow_tree(X_arr, y_arr, depth=0)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict continuous outputs for input samples.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        if self.tree_ is None:
            raise RuntimeError("Estimator has not been fitted.")

        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_arr.shape[1]}."
            )

        return np.array([self._traverse(x, self.tree_) for x in X_arr])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute R² (coefficient of determination).

        Returns
        -------
        float
        """
        y_true = _ensure_1d_float(y, "y")
        y_pred = self.predict(X)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)

    # -----------------------------------------------------------------
    # Tree construction
    # -----------------------------------------------------------------

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _RegressionNode:
        """
        Recursively build the tree from data (X, y).
        """
        n_samples = X.shape[0]
        node_value = float(np.mean(y))

        # Termination conditions
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.var(y) == 0
        ):
            return _RegressionNode(value=node_value)

        feat, thresh, left_mask, right_mask = self._best_split(X, y)

        if feat is None:
            return _RegressionNode(value=node_value)

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _RegressionNode(
            feature_index=feat,
            threshold=thresh,
            left=left_child,
            right=right_child,
            value=node_value,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], np.ndarray, np.ndarray]:
        """
        Identify the split that minimizes weighted variance.
        """
        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, np.array([]), np.array([])

        # Select candidate features
        if self.max_features is None:
            features = np.arange(n_features)
        elif isinstance(self.max_features, int):
            features = self._rng.choice(
                n_features, min(self.max_features, n_features), replace=False
            )
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
            features = self._rng.choice(n_features, k, replace=False)
        else:
            features = np.arange(n_features)

        best_mse = np.inf
        best_feat = None
        best_thresh = None
        best_left = best_right = np.array([], dtype=bool)

        base_var = np.var(y)

        for feat in features:
            col = X[:, feat]
            for t in np.unique(col):
                left = col <= t
                right = ~left

                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue

                mse = (
                    left.sum() * np.var(y[left]) +
                    right.sum() * np.var(y[right])
                ) / n_samples

                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = float(t)
                    best_left = left
                    best_right = right

        if best_feat is None or best_mse >= base_var:
            return None, None, np.array([]), np.array([])

        return best_feat, best_thresh, best_left, best_right

    # -----------------------------------------------------------------
    # Prediction helper
    # -----------------------------------------------------------------

    def _traverse(self, x: np.ndarray, node: _RegressionNode) -> float:
        """
        Follow decision rules until a leaf is reached.
        """
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
