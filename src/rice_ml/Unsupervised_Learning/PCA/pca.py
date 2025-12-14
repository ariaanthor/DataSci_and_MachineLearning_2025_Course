"""
Principal Component Analysis (NumPy-only).

This module provides PCA for dimensionality reduction.

Classes
-------
PCA
    Principal Component Analysis.

Examples
--------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.pca import PCA
>>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> pca = PCA(n_components=2)
>>> X_reduced = pca.fit_transform(X)
>>> X_reduced.shape
(3, 2)
"""

from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np

__all__ = ['PCA']

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
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


class PCA:
    """
    Principal Component Analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition (SVD)
    to project data to a lower-dimensional space.

    Parameters
    ----------
    n_components : int or float or None, default=None
        Number of components to keep:
        - int: exact number of components
        - float in (0, 1): select components to explain this fraction of variance
        - None: keep all components

    Attributes
    ----------
    n_components_ : int
        Number of components kept.
    components_ : ndarray of shape (n_components, n_features)
        Principal axes (eigenvectors).
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each component.
    singular_values_ : ndarray of shape (n_components,)
        Singular values corresponding to each component.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean estimated from training data.
    n_features_ : int
        Number of features in training data.
    n_samples_ : int
        Number of samples in training data.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    >>> pca = PCA(n_components=1)
    >>> X_reduced = pca.fit_transform(X)
    >>> X_reduced.shape
    (4, 1)
    >>> pca.explained_variance_ratio_[0] > 0.99
    True
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
    ) -> None:
        self.n_components = n_components

        self.n_components_: Optional[int] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None

    def fit(self, X: ArrayLike) -> "PCA":
        """
        Fit the PCA model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")
        n_samples, n_features = X_arr.shape

        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # Center the data
        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_

        # Perform SVD
        # X = U @ S @ Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Explained variance
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance

        # Determine number of components
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        elif isinstance(self.n_components, float):
            if not 0 < self.n_components < 1:
                raise ValueError(
                    "n_components as float must be in (0, 1) for variance ratio."
                )
            # Find components that explain desired variance
            cumsum = np.cumsum(explained_variance_ratio)
            n_components = np.searchsorted(cumsum, self.n_components) + 1
            n_components = min(n_components, min(n_samples, n_features))
        elif isinstance(self.n_components, int):
            if self.n_components < 1:
                raise ValueError("n_components must be >= 1.")
            n_components = min(self.n_components, min(n_samples, n_features))
        else:
            raise ValueError("n_components must be int, float, or None.")

        self.n_components_ = n_components
        self.components_ = Vt[:n_components]
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]
        self.singular_values_ = S[:n_components]

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Apply dimensionality reduction.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_components)
            Transformed data.
        """
        if self.components_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")

        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_}."
            )

        X_centered = X_arr - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform data back to original space.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_components)
            Data in reduced space.

        Returns
        -------
        X_original : ndarray, shape (n_samples, n_features)
            Data in original space.
        """
        if self.components_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")

        if X_arr.shape[1] != self.n_components_:
            raise ValueError(
                f"X has {X_arr.shape[1]} components, expected {self.n_components_}."
            )

        return X_arr @ self.components_ + self.mean_

    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the generative model.

        Returns
        -------
        cov : ndarray, shape (n_features, n_features)
            Estimated covariance matrix.
        """
        if self.components_ is None:
            raise RuntimeError("Model is not fitted.")

        # Cov = V @ diag(explained_variance) @ V.T
        return self.components_.T @ np.diag(self.explained_variance_) @ self.components_

    def score(self, X: ArrayLike) -> float:
        """
        Return the average log-likelihood of the samples.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        float
            Average log-likelihood (higher is better).
        """
        if self.components_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        n_samples = X_arr.shape[0]

        # Reconstruction error
        X_transformed = self.transform(X_arr)
        X_reconstructed = self.inverse_transform(X_transformed)
        reconstruction_error = np.sum((X_arr - X_reconstructed) ** 2) / n_samples

        # Use negative reconstruction error as score
        return -reconstruction_error