"""
community_detection.py

This module implements two classic community detection approaches for graphs
represented as adjacency/affinity matrices:

1) SpectralClustering
   - Builds a graph Laplacian from an affinity matrix (or an RBF kernel over
     feature vectors), embeds nodes using the smallest Laplacian eigenvectors,
     then clusters the embedding with a small NumPy K-Means implementation.

2) LabelPropagation
   - Iteratively updates node labels by taking a weighted majority vote over
     neighbors until convergence (or a max-iteration cap).

Notes
-----
- Graphs are represented as dense NumPy arrays.
- These implementations aim to be readable, testable, and dependency-free.

Classes
-------
SpectralClustering
LabelPropagation
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["SpectralClustering", "LabelPropagation"]

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------
def _as_2d_float(x: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert input to a non-empty 2D float NumPy array.

    Parameters
    ----------
    x : array_like
        Input matrix-like structure.
    name : str
        Name used in error messages.

    Returns
    -------
    ndarray
        A 2D float array.

    Raises
    ------
    ValueError
        If input is not 2D or is empty.
    TypeError
        If input contains non-numeric values that cannot be cast to float.
    """
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
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


# ---------------------------------------------------------------------
# Spectral clustering
# ---------------------------------------------------------------------
class SpectralClustering:
    """
    Spectral clustering for graphs (and optionally feature vectors).

    The typical workflow is:
      - Build an affinity matrix W (either precomputed or via an RBF kernel)
      - Build the normalized Laplacian L = I - D^{-1/2} W D^{-1/2}
      - Compute the smallest-eigenvector embedding of L
      - Run K-Means on the row-normalized embedding

    Parameters
    ----------
    n_clusters : int, default=2
        Number of communities to find.
    affinity : {"precomputed", "rbf"}, default="precomputed"
        - "precomputed": X is an (n, n) affinity/adjacency matrix.
        - "rbf": X is an (n, d) feature matrix; affinity computed with an RBF kernel.
    gamma : float or None, default=None
        RBF kernel coefficient. If None, uses 1 / n_features.
    n_init : int, default=10
        Number of random restarts for K-Means.
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels after fitting.
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        The affinity matrix used during fitting.

    Examples
    --------
    Cluster two groups using a precomputed adjacency matrix:

    >>> import numpy as np
    >>> W = np.array([
    ...     [0, 1, 1, 0],
    ...     [1, 0, 1, 0],
    ...     [1, 1, 0, 0],
    ...     [0, 0, 0, 0],
    ... ], dtype=float)
    >>> model = SpectralClustering(n_clusters=2, random_state=1)
    >>> out = model.fit_predict(W)
    >>> out.shape
    (4,)
    """

    def __init__(
        self,
        n_clusters: int = 2,
        affinity: str = "precomputed",
        gamma: Optional[float] = None,
        n_init: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if affinity not in ("precomputed", "rbf"):
            raise ValueError("affinity must be 'precomputed' or 'rbf'.")
        if n_init < 1:
            raise ValueError("n_init must be >= 1.")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be positive when provided.")

        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_init = n_init
        self.random_state = random_state

        self.labels_: Optional[np.ndarray] = None
        self.affinity_matrix_: Optional[np.ndarray] = None

    def _affinity_from_input(self, X: np.ndarray) -> np.ndarray:
        """Create an affinity matrix based on the configured affinity strategy."""
        if self.affinity == "precomputed":
            return X

        # RBF kernel over features:
        # W_ij = exp(-gamma * ||x_i - x_j||^2)
        g = float(self.gamma) if self.gamma is not None else (1.0 / X.shape[1])

        # Efficient squared-distance matrix:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        sq_norms = np.sum(X * X, axis=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
        sq_dists = np.maximum(sq_dists, 0.0)  # guard against tiny negatives from FP error
        return np.exp(-g * sq_dists)

    def _normalized_laplacian(self, W: np.ndarray) -> np.ndarray:
        """
        Compute the normalized graph Laplacian: L = I - D^{-1/2} W D^{-1/2}.
        """
        deg = np.sum(W, axis=1)
        inv_sqrt_deg = np.where(deg > 0.0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        I = np.eye(W.shape[0], dtype=float)
        return I - (D_inv_sqrt @ W @ D_inv_sqrt)

    def _kmeans(
        self, X: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 100
    ) -> np.ndarray:
        """
        Minimal K-Means (squared Euclidean) with multiple restarts.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        n = X.shape[0]
        best_labels: Optional[np.ndarray] = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            # Randomly pick k distinct points as initial centers.
            init_idx = rng.choice(n, size=k, replace=False)
            centers = X[init_idx].copy()

            for _ in range(max_iter):
                # Assignment step
                # dist(i, c) = ||x_i - center_c||^2
                d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                labels = np.argmin(d2, axis=1)

                # Update step
                new_centers = centers.copy()
                for c in range(k):
                    mask = labels == c
                    if np.any(mask):
                        new_centers[c] = X[mask].mean(axis=0)
                    # else: keep center as-is (empty cluster)

                if np.allclose(new_centers, centers):
                    centers = new_centers
                    break
                centers = new_centers

            # Inertia = sum_i min_c ||x_i - center_c||^2
            d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            inertia = float(np.sum(np.min(d2, axis=1)))

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels

        # best_labels is always set because n_init >= 1
        return best_labels  # type: ignore[return-value]

    def fit(self, X: ArrayLike) -> "SpectralClustering":
        """
        Fit the clustering model.

        Parameters
        ----------
        X : array_like
            If affinity="precomputed": shape (n, n) adjacency/affinity matrix.
            If affinity="rbf": shape (n, d) feature matrix.

        Returns
        -------
        self
        """
        X_arr = _as_2d_float(X, "X")

        if self.affinity == "precomputed" and X_arr.shape[0] != X_arr.shape[1]:
            raise ValueError("For affinity='precomputed', X must be square (n, n).")

        # Step 1: affinity matrix
        W = self._affinity_from_input(X_arr)
        self.affinity_matrix_ = W

        # Step 2: Laplacian + eigendecomposition
        L = self._normalized_laplacian(W)
        _, eigvecs = np.linalg.eigh(L)

        # Step 3: embedding = k smallest-eigenvalue eigenvectors
        emb = eigvecs[:, : self.n_clusters]

        # Row-normalize the embedding (classic spectral clustering trick)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        emb = emb / norms

        # Step 4: K-Means over embedding
        rng = np.random.default_rng(self.random_state)
        self.labels_ = self._kmeans(emb, self.n_clusters, rng)
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Fit and return cluster labels.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        self.fit(X)
        return self.labels_


# ---------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------
class LabelPropagation:
    """
    Label propagation for community detection (weighted adjacency).

    Each node starts with a unique label. Repeatedly, nodes update their label
    to the label with the largest *total edge weight* among its neighbors.
    Ties are broken randomly using `random_state`.

    Parameters
    ----------
    max_iter : int, default=30
        Maximum number of update sweeps.
    random_state : int or None, default=None
        Seed for tie-breaking and node update order.

    Attributes
    ----------
    labels_ : ndarray of shape (n_nodes,)
        Final labels (re-indexed to consecutive integers 0..K-1).
    n_iter_ : int
        Number of iterations performed.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([
    ...     [0, 1, 0, 0],
    ...     [1, 0, 1, 0],
    ...     [0, 1, 0, 1],
    ...     [0, 0, 1, 0],
    ... ], dtype=float)
    >>> lp = LabelPropagation(max_iter=50, random_state=0)
    >>> labs = lp.fit_predict(A)
    >>> labs.shape
    (4,)
    """

    def __init__(self, max_iter: int = 30, random_state: Optional[int] = None) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        self.max_iter = max_iter
        self.random_state = random_state

        self.labels_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    def fit(self, X: ArrayLike) -> "LabelPropagation":
        """
        Run label propagation on a graph adjacency matrix.

        Parameters
        ----------
        X : array_like, shape (n_nodes, n_nodes)
            Weighted adjacency matrix. Non-edges should be 0.

        Returns
        -------
        self
        """
        A = _as_2d_float(X, "X")
        if A.shape[0] != A.shape[1]:
            raise ValueError("X must be a square adjacency matrix (n_nodes, n_nodes).")

        n = A.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Start with each node in its own community.
        labels = np.arange(n)

        for it in range(self.max_iter):
            self.n_iter_ = it + 1
            prev = labels.copy()

            # Update nodes in a random order (helps avoid cyclic behavior).
            for i in rng.permutation(n):
                nbrs = np.flatnonzero(A[i] > 0.0)
                if nbrs.size == 0:
                    continue  # isolated node, keep its label

                # Weighted vote: accumulate weights per neighbor label.
                nbr_labels = labels[nbrs]
                nbr_weights = A[i, nbrs]

                uniq = np.unique(nbr_labels)
                scores = np.zeros(uniq.size, dtype=float)
                for j, lab in enumerate(uniq):
                    scores[j] = float(np.sum(nbr_weights[nbr_labels == lab]))

                max_score = np.max(scores)
                best = uniq[scores == max_score]

                # Tie-break randomly if needed.
                labels[i] = rng.choice(best) if best.size > 1 else best[0]

            # Converged: no labels changed in this sweep.
            if np.array_equal(labels, prev):
                break

        # Compress labels to 0..K-1 for cleaner output
        _, inv = np.unique(labels, return_inverse=True)
        self.labels_ = inv
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Fit and return community labels.
        """
        self.fit(X)
        return self.labels_
