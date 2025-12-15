"""
dbscan.py

This module implements **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)
using only NumPy. DBSCAN discovers clusters as **connected dense regions** in the data and
labels points that do not belong to any cluster as **noise** (-1).

Compared to centroid-based methods (e.g., k-means), DBSCAN:
- does **not** require specifying the number of clusters in advance,
- can find **non-spherical** clusters,
- explicitly identifies **outliers**.

Notes
-----
This implementation precomputes a full pairwise distance matrix, so it is best suited for
small-to-medium datasets.

Classes
-------
DBSCAN
    Density-based clustering with noise labeling.

Quick Example
-------------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.dbscan import DBSCAN
>>> X = np.array([[0., 0.], [0.2, 0.1], [0.1, 0.3], [5., 5.], [5.1, 4.9]], dtype=float)
>>> model = DBSCAN(eps=0.5, min_samples=2).fit(X)
>>> set(model.labels_)
{0, 1}
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union
import numpy as np

__all__ = ["DBSCAN"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _as_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert input to a non-empty 2D float ndarray.

    Parameters
    ----------
    X : array_like
        Input data. If 1D, it is interpreted as (n_samples, 1).
    name : str
        Name used for error messages.

    Returns
    -------
    np.ndarray
        2D array of dtype float.

    Raises
    ------
    ValueError
        If X is empty or not 1D/2D.
    TypeError
        If X contains non-numeric values.
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


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Parameters
    ----------
    eps : float, default=0.5
        Neighborhood radius. Two points are considered neighbors if their distance
        is <= eps.
    min_samples : int, default=5
        Minimum number of points (including the point itself) within eps to qualify
        as a **core** point.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric used to build neighborhoods.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point. Noise points have label -1.
    core_sample_indices_ : ndarray
        Indices of core samples in the training data.
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample.
    n_features_ : int
        Number of features in the fitted data.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [1.1, 1.0], [10, 10], [10.2, 9.9], [50, 50]], dtype=float)
    >>> db = DBSCAN(eps=0.5, min_samples=2)
    >>> labels = db.fit_predict(X)
    >>> labels.tolist()
    [0, 0, 1, 1, -1]
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
    ) -> None:
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite number.")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")

        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = metric

        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    # ------------------------- distance + neighborhoods -------------------------

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the full pairwise distance matrix.

        Returns
        -------
        D : ndarray of shape (n_samples, n_samples)
            D[i, j] is the distance between X[i] and X[j].
        """
        n = X.shape[0]

        if self.metric == "euclidean":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b  (vectorized)
            sq = np.sum(X * X, axis=1)
            D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
            # small negative values can appear due to floating point; clamp to 0
            D2 = np.maximum(D2, 0.0)
            return np.sqrt(D2)

        # Manhattan / L1: sum_k |x_i[k] - x_j[k]|
        # Broadcasting yields (n, n, d) -> reduce on last axis.
        # This is clear but may be memory-heavy for large n.
        return np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=2)

    def _neighbors_from_dist(self, D: np.ndarray, i: int) -> np.ndarray:
        """Indices of all points within eps of point i (including i)."""
        return np.flatnonzero(D[i] <= self.eps)

    # ------------------------------ core algorithm ------------------------------

    def fit(self, X: ArrayLike) -> "DBSCAN":
        """
        Fit DBSCAN on X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training samples.

        Returns
        -------
        self
        """
        X_arr = _as_2d_float(X, "X")
        n_samples, n_features = X_arr.shape
        self.n_features_ = n_features

        # 1) Precompute distances and neighborhood sizes
        D = self._pairwise_distances(X_arr)
        neighborhood_sizes = np.sum(D <= self.eps, axis=1)

        # Core points have enough neighbors (including themselves)
        core_mask = neighborhood_sizes >= self.min_samples
        self.core_sample_indices_ = np.flatnonzero(core_mask)
        self.components_ = X_arr[self.core_sample_indices_].copy()

        # 2) Cluster expansion
        labels = np.full(n_samples, -1, dtype=int)  # -1 means "unassigned/noise"
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            nbrs = self._neighbors_from_dist(D, i)

            # If i is not a core point, we cannot expand a cluster from it.
            if neighborhood_sizes[i] < self.min_samples:
                # leave label as -1 (could be reassigned later as a border point)
                continue

            # Start a new cluster rooted at core point i
            labels[i] = cluster_id

            # Use a simple queue to expand reachable density-connected points
            queue = list(nbrs)
            q_idx = 0
            while q_idx < len(queue):
                j = queue[q_idx]
                q_idx += 1

                if not visited[j]:
                    visited[j] = True
                    j_nbrs = self._neighbors_from_dist(D, j)

                    # Only core points "grow" the frontier
                    if neighborhood_sizes[j] >= self.min_samples:
                        # Add new candidates (duplicates are okay; visited handles it)
                        queue.extend(j_nbrs.tolist())

                # Assign to cluster if not already assigned
                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Fit DBSCAN and return cluster labels.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noise points are -1.
        """
        self.fit(X)
        return self.labels_
