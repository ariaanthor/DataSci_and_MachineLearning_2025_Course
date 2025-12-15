"""
knn.py

Classification & Regression implementations of K-Nearest Neighbors.
The goal is to keep the code compact and readable while
still being robust to common user input mistakes.

Implemented models
------------------
KNNClassifier
    Discrete label prediction with optional probability estimates.

KNNRegressor
    Continuous target prediction via (weighted) averaging.

Supported options
-----------------
- Distance metrics: "euclidean", "manhattan"
- Neighbor weighting:
    * "uniform": each neighbor counts equally
    * "distance": closer neighbors contribute more (inverse-distance), with
      exact matches handled specially

Small demo
----------
>>> import numpy as np
>>> from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor
>>> Xc = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.]])
>>> yc = np.array([0, 0, 1, 1])
>>> KNNClassifier(n_neighbors=1).fit(Xc, yc).predict([[1.9, 0.1]]).tolist()
[1]
>>>
>>> Xr = np.array([[0.], [1.], [4.]], dtype=float)
>>> yr = np.array([0.0, 1.0, 2.0])
>>> float(KNNRegressor(n_neighbors=2, weights="uniform").fit(Xr, yr).predict([[2.0]])[0])
0.5
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple, Union, Sequence

import numpy as np

__all__ = [
    "KNNClassifier",
    "KNNRegressor",
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ==============================
# Input utilities / validation
# ==============================

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert input into a 2D float array.

    This file treats all feature data as a matrix shaped (n_samples, n_features).
    To keep that invariant:
    - we reject non-2D inputs,
    - enforce non-empty arrays,
    - and coerce to float when possible.

    Parameters
    ----------
    X : ArrayLike
        Candidate feature matrix.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        2D array (float dtype).

    Raises
    ------
    ValueError
        If `X` is not 2D or is empty.
    TypeError
        If numeric conversion fails.
    """
    arr = np.asarray(X)

    # KNN in this implementation expects a design matrix, not a vector
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    # Force float representation; complain if not possible
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)

    return arr


def _ensure_1d(y, name: str = "y") -> np.ndarray:
    """
    Ensure y is a 1D, non-empty array.

    Labels for classification may be any dtype; regression targets are checked
    separately in the regressor.

    Parameters
    ----------
    y : array_like
        Candidate label/target vector.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        1D array view/copy.

    Raises
    ------
    ValueError
        If `y` is not 1D or empty.
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    """
    Helper to produce a NumPy Generator from a seed-like input.
    """
    if seed is None:
        return np.random.default_rng()
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(seed))


def _validate_common_params(
    n_neighbors: int,
    metric: Literal["euclidean", "manhattan"],
    weights: Literal["uniform", "distance"],
) -> None:
    """
    Validate hyperparameters shared by classifier and regressor.
    """
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    if metric not in ("euclidean", "manhattan"):
        raise ValueError("metric must be 'euclidean' or 'manhattan'.")
    if weights not in ("uniform", "distance"):
        raise ValueError("weights must be 'uniform' or 'distance'.")


# ==============================
# Distance + neighbor selection
# ==============================

def _pairwise_distances(XA: np.ndarray, XB: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute distances between every row in XA and every row in XB.

    Returns a matrix D where D[i, j] is the distance between XA[i] and XB[j].

    Implementation notes
    --------------------
    - Euclidean distance uses a quadratic expansion to avoid explicit broadcasting.
    - Manhattan distance uses broadcasting and absolute differences.
    """
    if metric == "euclidean":
        # Use ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b, then sqrt
        aa = np.sum(XA * XA, axis=1, keepdims=True)          # (n_a, 1)
        bb = np.sum(XB * XB, axis=1, keepdims=True).T        # (1, n_b)
        D2 = np.maximum(aa + bb - 2.0 * XA @ XB.T, 0.0)      # numerical guard
        return np.sqrt(D2, dtype=float)
    elif metric == "manhattan":
        # Broadcast to (n_a, n_b, d) and reduce over d
        diff = XA[:, None, :] - XB[None, :, :]
        return np.sum(np.abs(diff), axis=2, dtype=float)
    else:
        raise ValueError("Unsupported metric.")


def _neighbors(
    X_train: np.ndarray,
    X_query: np.ndarray,
    n_neighbors: int,
    metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify k nearest neighbors in X_train for each row of X_query.

    Returns
    -------
    distances : ndarray, shape (n_query, k)
        Sorted distances for each query sample.
    indices : ndarray, shape (n_query, k)
        Training indices corresponding to those distances.
    """
    # Compute full distance matrix once
    D = _pairwise_distances(X_query, X_train, metric)

    if n_neighbors > X_train.shape[0]:
        raise ValueError(
            f"n_neighbors={n_neighbors} cannot exceed number of training samples={X_train.shape[0]}."
        )

    # Fast selection: grab k smallest entries per row without sorting everything
    idx = np.argpartition(D, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]

    # Now sort the selected k neighbors by their actual distances
    row = np.arange(D.shape[0])[:, None]
    dsel = D[row, idx]
    order = np.argsort(dsel, axis=1)

    idx_sorted = idx[row, order]
    d_sorted = dsel[row, order]
    return d_sorted, idx_sorted


def _weights_from_distances(dist: np.ndarray, scheme: str, eps: float = 1e-12) -> np.ndarray:
    """
    Build a weight matrix for neighbor contributions.

    Weight schemes
    --------------
    uniform
        All neighbors get weight 1.
    distance
        Use inverse distance 1/d. If any neighbor has distance ~0 for a query,
        then only those exact-match neighbors receive weight 1 and the rest get 0.

    Parameters
    ----------
    dist : ndarray, shape (n_query, k)
        Neighbor distances.
    scheme : {"uniform", "distance"}
        Weighting rule.
    eps : float
        Small constant to avoid division-by-zero.

    Returns
    -------
    ndarray
        Nonnegative (unnormalized) weights of shape (n_query, k).
    """
    if scheme == "uniform":
        return np.ones_like(dist, dtype=float)

    # Distance weighting
    zero_mask = (dist <= eps)
    w = np.empty_like(dist, dtype=float)

    # If we have any exact matches, restrict voting/averaging to those matches
    any_zero = zero_mask.any(axis=1)
    if np.any(any_zero):
        w[any_zero] = zero_mask[any_zero].astype(float)

    # Otherwise, use inverse distances
    if np.any(~any_zero):
        w[~any_zero] = 1.0 / np.maximum(dist[~any_zero], eps)

    return w


# ==============================
# Base estimator (shared logic)
# ==============================

class _KNNBase:
    """
    Shared mechanics for KNN classifier/regressor.

    The "model" here is just stored training data; prediction is performed
    by searching for nearest neighbors at query time.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        _validate_common_params(n_neighbors, metric, weights)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.weights = weights

        # Populated by fit()
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Store the training set.

        KNN has no parameter learning; "fitting" just validates input shapes
        and caches them for later neighbor queries.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if len(y_arr) != X_arr.shape[0]:
            raise ValueError(
                f"X and y length mismatch: len(y)={len(y_arr)} vs X.shape[0]={X_arr.shape[0]}"
            )

        if self.n_neighbors > X_arr.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")

        self._X = X_arr
        self._y = y_arr
        return self

    def _check_is_fitted(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Centralized guard: ensure fit() has been called.
        """
        if self._X is None or self._y is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return self._X, self._y

    def kneighbors(self, X: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return distances + indices of the k nearest training samples.

        This is useful for debugging and for building custom post-processing on top
        of KNN outputs.
        """
        X_train, _ = self._check_is_fitted()
        Xq = _ensure_2d_float(X, "X")

        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")

        return _neighbors(X_train, Xq, self.n_neighbors, self.metric)


# ==============================
# Classification
# ==============================

class KNNClassifier(_KNNBase):
    """
    k-NN classifier.

    Produces hard labels via majority vote, and probability estimates via
    normalized vote totals.

    Example
    -------
    >>> import numpy as np
    >>> X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    >>> y = np.array(["red", "red", "blue", "blue"], dtype=object)
    >>> knn = KNNClassifier(n_neighbors=3, metric="manhattan").fit(X, y)
    >>> knn.predict([[0.0, 0.9], [1.0, 0.1]]).tolist()
    ['red', 'blue']
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        super().__init__(n_neighbors=n_neighbors, metric=metric, weights=weights)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        super().fit(X, y)

        # Keep a fixed ordering of class labels for proba output columns
        self.classes_ = np.unique(self._y)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Estimate class probabilities by aggregating neighbor votes.

        Returns a matrix of shape (n_query, n_classes), where each row sums to 1.
        The column ordering matches `self.classes_`.
        """
        X_train, y_train = self._check_is_fitted()
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")

        Xq = _ensure_2d_float(X, "X")
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")

        dist, idx = _neighbors(X_train, Xq, self.n_neighbors, self.metric)
        w = _weights_from_distances(dist, self.weights)

        n_query = Xq.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_query, n_classes), dtype=float)

        # Per-query voting; bincount provides an efficient "sum weights per class"
        for i in range(n_query):
            neigh_labels = y_train[idx[i]]
            class_ids = np.searchsorted(self.classes_, neigh_labels)
            counts = np.bincount(class_ids, weights=w[i], minlength=n_classes)
            total = counts.sum()
            proba[i] = (counts / total) if total != 0 else (1.0 / n_classes)

        return proba

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict the most likely class for each query point.
        """
        proba = self.predict_proba(X)
        best = np.argmax(proba, axis=1)
        return self.classes_[best]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return accuracy (fraction correct) on a labeled dataset.
        """
        y_true = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        if len(y_true) != len(y_pred):
            raise ValueError("X and y lengths do not match.")
        return float(np.mean(y_true == y_pred))


# ==============================
# Regression
# ==============================

class KNNRegressor(_KNNBase):
    """
    k-NN regressor.

    The prediction for a query point is an average of neighbor targets:
    - uniform weights -> simple mean
    - distance weights -> inverse-distance weighted mean

    Example
    -------
    >>> import numpy as np
    >>> X = np.array([[0.], [2.], [3.]], dtype=float)
    >>> y = np.array([0.0, 2.0, 3.0])
    >>> reg = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
    >>> round(float(reg.predict([[2.5]])[0]), 6)
    2.5
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNRegressor":
        """
        Cache the training data, enforcing numeric targets.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        # Regression requires numeric targets (coerce if possible)
        if not np.issubdtype(y_arr.dtype, np.number):
            try:
                y_arr = y_arr.astype(float, copy=False)
            except (TypeError, ValueError) as e:
                raise TypeError("Regression target values must be numeric.") from e

        if len(y_arr) != X_arr.shape[0]:
            raise ValueError(
                f"X and y length mismatch: len(y)={len(y_arr)} vs X.shape[0]={X_arr.shape[0]}"
            )

        if self.n_neighbors > X_arr.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")

        self._X = X_arr
        self._y = y_arr.astype(float, copy=False)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict continuous values for query inputs.

        Returns
        -------
        np.ndarray
            Float predictions, shape (n_query,).
        """
        X_train, y_train = self._check_is_fitted()
        Xq = _ensure_2d_float(X, "X")

        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")

        dist, idx = _neighbors(X_train, Xq, self.n_neighbors, self.metric)
        w = _weights_from_distances(dist, self.weights)

        # Gather neighbor targets into (n_query, k)
        y_neighbors = y_train[idx]

        # Weighted mean per query
        wsum = np.sum(w, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            y_pred = np.divide(np.sum(w * y_neighbors, axis=1), wsum, where=wsum != 0)

        # If a row has all weights zero (should be rare), fall back to unweighted mean
        fallback = (wsum == 0)
        if np.any(fallback):
            y_pred[fallback] = np.mean(y_neighbors[fallback], axis=1)

        return y_pred.astype(float, copy=False)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute R^2 (coefficient of determination) on (X, y).

        This method follows the standard definition:
            R^2 = 1 - SS_res / SS_tot

        A special-case guard is included for constant y_true, mirroring the
        behavior described in the original implementation.
        """
        X_train, _ = self._check_is_fitted()
        Xq = _ensure_2d_float(X, "X")
        y_true = np.asarray(_ensure_1d(y, "y"), dtype=float)

        if Xq.shape[0] != y_true.shape[0]:
            raise ValueError("X and y lengths do not match.")

        y_pred = self.predict(Xq)

        ss_res = np.sum((y_true - y_pred) ** 2)
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)

        if ss_tot == 0:
            # When y_true has zero variance, R^2 is not meaningful unless the fit is perfect
            if np.array_equal(Xq, X_train) and ss_res == 0:
                return 1.0
            raise ValueError(
                "R^2 is undefined when y_true is constant unless scoring on the "
                "training inputs with a perfect fit."
            )

        return float(1.0 - ss_res / ss_tot)
