"""
k_means_clustering.py

This module implements the classic **K-Means** algorithm for partitioning data into
`k` clusters by minimizing the within-cluster sum of squared distances (inertia).
It supports two common centroid initialization strategies (random and k-means++)
and repeats clustering multiple times (`n_init`) to reduce sensitivity to bad
initializations.

Notes
-----
- Distance computations use squared Euclidean distance for efficiency.
- This implementation is intended for educational / small-to-medium workloads.

Classes
-------
KMeans
    K-Means clustering estimator.

Quick Example
-------------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.k_means_clustering import KMeans
>>> X = np.array([[0., 0.], [0., 1.], [9., 9.], [10., 9.]], dtype=float)
>>> km = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=0).fit(X)
>>> km.cluster_centers_.shape
(2, 2)
>>> km.predict([[0.2, 0.2], [9.5, 9.2]]).tolist()
[km.labels_[0], km.labels_[2]]
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union, Tuple
import numpy as np

__all__ = ["KMeans"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _as_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert input into a non-empty 2D float array.

    Parameters
    ----------
    X : array_like
        If 1D, it is reshaped to (n_samples, 1).
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        2D array of dtype float.

    Raises
    ------
    ValueError
        If X is empty or not 1D/2D.
    TypeError
        If X contains non-numeric data.
    """
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


class KMeans:
    """
    K-Means clustering estimator.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form.
    init : {"k-means++", "random"}, default="k-means++"
        Centroid initialization strategy.
    n_init : int, default=10
        Number of independent runs; the best (lowest inertia) is kept.
    max_iter : int, default=300
        Maximum number of iterations per run.
    tol : float, default=1e-4
        Convergence threshold on centroid movement (squared L2).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Cluster assignment for each training sample.
    inertia_ : float
        Sum of squared distances to the closest centroid.
    n_iter_ : int
        Iterations used in the best run.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1., 2.], [1., 4.], [10., 2.], [10., 4.]], dtype=float)
    >>> km = KMeans(n_clusters=2, random_state=1).fit(X)
    >>> sorted(set(km.labels_.tolist()))
    [0, 1]
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'.")
        if n_init < 1:
            raise ValueError("n_init must be >= 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if tol < 0:
            raise ValueError("tol must be >= 0.")

        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    # ----------------------------- init methods -----------------------------

    def _init_random(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Pick k distinct points as the initial centroids."""
        n = X.shape[0]
        picks = rng.choice(n, size=self.n_clusters, replace=False)
        return X[picks].copy()

    def _init_kmeanspp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        k-means++ initialization.

        Select the first centroid uniformly at random. Each subsequent centroid is
        sampled with probability proportional to the squared distance to the closest
        already-chosen centroid.
        """
        n, d = X.shape
        centers = np.empty((self.n_clusters, d), dtype=float)

        first = rng.integers(0, n)
        centers[0] = X[first]

        # Keep track of the closest squared distance to any chosen center
        closest_d2 = np.sum((X - centers[0]) ** 2, axis=1)

        for k in range(1, self.n_clusters):
            total = float(np.sum(closest_d2))
            if total == 0.0:
                # All points are identical (or numerically identical)
                centers[k:] = centers[k - 1]
                break

            probs = closest_d2 / total
            idx = rng.choice(n, p=probs)
            centers[k] = X[idx]

            # Update closest distances with the new center
            new_d2 = np.sum((X - centers[k]) ** 2, axis=1)
            closest_d2 = np.minimum(closest_d2, new_d2)

        return centers

    # -------------------------- core computations ---------------------------

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Return squared Euclidean distances from each sample to each center.

        Output shape: (n_samples, n_clusters)
        """
        return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _assign(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign each sample to its closest center."""
        return np.argmin(self._squared_distances(X, centers), axis=1)

    def _inertia(self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
        """Compute sum of squared distances to each sample's assigned centroid."""
        d2 = self._squared_distances(X, centers)
        return float(np.sum(d2[np.arange(labels.size), labels]))

    def _recompute_centers(self, X: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Update cluster centers as the mean of assigned points.

        If a cluster becomes empty, re-seed it to a random data point to keep
        `n_clusters` stable.
        """
        n_features = X.shape[1]
        centers = np.empty((self.n_clusters, n_features), dtype=float)

        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize to a random point (common practical fix)
                centers[k] = X[rng.integers(0, X.shape[0])]
        return centers

    def _run_once(self, X: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Single K-Means run; returns (centers, labels, inertia, n_iter)."""
        if self.init == "k-means++":
            centers = self._init_kmeanspp(X, rng)
        else:
            centers = self._init_random(X, rng)

        n_iter = 0
        for it in range(self.max_iter):
            n_iter = it + 1

            labels = self._assign(X, centers)
            new_centers = self._recompute_centers(X, labels, rng)

            shift = float(np.sum((new_centers - centers) ** 2))
            centers = new_centers

            # tol is a threshold on centroid movement (squared) across all centers
            if shift <= self.tol:
                break

        labels = self._assign(X, centers)
        inertia = self._inertia(X, labels, centers)
        return centers, labels, inertia, n_iter

    # ------------------------------ public API ------------------------------

    def fit(self, X: ArrayLike) -> "KMeans":
        """
        Fit the model to X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        X_arr = _as_2d_float(X, "X")
        if X_arr.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X_arr.shape[0]} must be >= n_clusters={self.n_clusters}."
            )

        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_iters = 0

        # Note: We reuse the same RNG stream across runs for reproducibility.
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._run_once(X_arr, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_iters = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_iters)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = _as_2d_float(X, "X")
        if X_arr.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.cluster_centers_.shape[1]}."
            )
        return self._assign(X_arr, self.cluster_centers_)

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Fit the model and return training labels.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        self.fit(X)
        return self.labels_

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Compute distances to cluster centers.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        distances : ndarray of shape (n_samples, n_clusters)
            Euclidean distances to each centroid.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = _as_2d_float(X, "X")
        if X_arr.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.cluster_centers_.shape[1]}."
            )
        return np.sqrt(self._squared_distances(X_arr, self.cluster_centers_))

    def score(self, X: ArrayLike) -> float:
        """
        Return negative inertia on X (sklearn-style).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        float
            -inertia(X)
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = _as_2d_float(X, "X")
        labels = self.predict(X_arr)
        return -self._inertia(X_arr, labels, self.cluster_centers_)
