"""
Perceptron for binary classification.

This file contains a minimal, educational implementation of the classic
single-layer perceptron algorithm. The model learns a linear decision
boundary by iterating through samples and updating weights only when a
point is misclassified.

What you get
------------
- A `Perceptron` class with a scikit-learn-like API: `fit`, `predict`, `score`
- Simple input validation helpers (`_ensure_2d_float`, `_ensure_1d`)
- Optional bias term (intercept) and basic early stopping

Quick demo
----------
Train on a toy linearly-separable dataset:

>>> import numpy as np
>>> from rice_ml.supervised_learning.perceptron import Perceptron
>>> X = np.array([[1, 1], [2, 0], [0, 2], [2, 2]], dtype=float)
>>> y = np.array(["neg", "neg", "neg", "pos"], dtype=object)
>>> clf = Perceptron(learning_rate=1.0, max_iter=50, random_state=0).fit(X, y)
>>> clf.predict([[2, 2]]).tolist()
['neg']
>>> 0.0 <= clf.score(X, y) <= 1.0
True
"""

from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np

__all__ = ["Perceptron"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Coerce feature input into a 2D float array.

    Behavior notes
    --------------
    - 1D input is interpreted as a single feature column (n_samples, 1).
    - Empty inputs are rejected.
    - Non-numeric dtypes are cast to float when possible.

    Parameters
    ----------
    X : array_like
        Feature matrix or vector.
    name : str
        Used in error messages.

    Returns
    -------
    arr : ndarray of shape (n_samples, n_features)
        Float feature matrix.

    Raises
    ------
    ValueError
        If X cannot be interpreted as 1D/2D or is empty.
    TypeError
        If elements cannot be converted to floats.
    """
    arr = np.asarray(X)

    # If the user passed a flat vector, treat it as one feature.
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    # Enforce numeric values (cast if needed).
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)

    return arr


def _ensure_1d(y: ArrayLike, name: str = "y") -> np.ndarray:
    """
    Ensure label/target input is a non-empty 1D array.

    Parameters
    ----------
    y : array_like
        Label vector.
    name : str
        Used in error messages.

    Returns
    -------
    arr : ndarray of shape (n_samples,)
        1D array view/copy of y.

    Raises
    ------
    ValueError
        If y is not 1D or is empty.
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


class Perceptron:
    """
    Binary perceptron classifier.

    The perceptron learns weights `w` (and optional bias `b`) using an
    online update. When a point is misclassified, we nudge the parameters
    in the direction that would correctly classify it.

    Update rule (in the internal {-1, +1} coding)
    ---------------------------------------------
    If sign(w·x + b) != y:
        w <- w + eta * y * x
        b <- b + eta * y        (only when fit_intercept=True)

    Parameters
    ----------
    learning_rate : float, default=1.0
        Step size used during updates.
    max_iter : int, default=1000
        Maximum number of epochs (full passes over the training set).
    tol : float or None, default=1e-3
        Optional early-stop threshold based on improvement in the number
        of mistakes between epochs. If None, the tolerance-based stop is
        disabled (but exact zero-error stop still applies).
    fit_intercept : bool, default=True
        Whether to learn a bias term.
    random_state : int or None, default=None
        Seed used for the epoch-wise shuffling.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Sorted unique labels observed during fitting.
    coef_ : ndarray of shape (n_features,)
        Weight vector (learned).
    intercept_ : float
        Bias term (learned; kept at 0.0 if fit_intercept=False).
    n_iter_ : int
        Number of epochs actually run.
    errors_ : list[int]
        Misclassification count per epoch.

    Examples
    --------
    A tiny separable dataset with string labels:

    >>> import numpy as np
    >>> X = np.array([[0., 0.], [0., 1.], [2., 0.], [2., 1.]])
    >>> y = np.array(["left", "left", "right", "right"], dtype=object)
    >>> p = Perceptron(max_iter=25, learning_rate=0.5, random_state=123).fit(X, y)
    >>> p.predict([[1.9, 0.2]]).tolist()
    ['right']
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 1000,
        tol: Optional[float] = 1e-3,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        # Basic sanity checks to avoid silent weirdness.
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        # Learned during fitting
        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0
        self.errors_: list = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> "Perceptron":
        """
        Learn parameters from training data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training features.
        y : array_like, shape (n_samples,)
            Training labels (must contain exactly two distinct values).

        Returns
        -------
        self : Perceptron
            The fitted model.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        # Identify the two labels (kept for mapping predictions back).
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(
                "Perceptron is a binary classifier; y must have exactly 2 classes."
            )

        n_samples, n_features = X_arr.shape

        # Internally: map labels to {-1, +1}, where classes_[1] is the +1 class.
        positive = self.classes_[1]
        y_binary = np.where(y_arr == positive, 1.0, -1.0)

        # Start from zero weights (classic perceptron baseline).
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0

        rng = np.random.default_rng(self.random_state)
        self.errors_ = []
        prev_errors = float("inf")

        for epoch in range(self.max_iter):
            # Shuffle each epoch so training isn't sensitive to ordering.
            order = rng.permutation(n_samples)
            X_epoch = X_arr[order]
            y_epoch = y_binary[order]

            mistakes = 0

            for xi, yi in zip(X_epoch, y_epoch):
                # Score and hard threshold at 0.
                score = xi @ self.coef_ + self.intercept_
                pred = 1 if score >= 0 else -1

                if yi != pred:
                    # Single-sample update.
                    step = self.learning_rate * yi
                    self.coef_ += step * xi
                    if self.fit_intercept:
                        self.intercept_ += step
                    mistakes += 1

            self.errors_.append(mistakes)
            self.n_iter_ = epoch + 1

            # Stop immediately if training set has no mistakes.
            if mistakes == 0:
                break

            # Optional "small improvement" early stopping.
            if self.tol is not None and (prev_errors - mistakes) < self.tol:
                break

            prev_errors = mistakes

        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """
        Compute raw (signed) scores before applying the class threshold.

        Interpretation
        --------------
        Positive scores correspond to `classes_[1]`, negative scores correspond
        to `classes_[0]`.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Linear scores X @ w + b.
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != len(self.coef_):
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {len(self.coef_)}."
            )

        return X_arr @ self.coef_ + self.intercept_

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict labels for new samples.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels (same dtype as the original y seen in fit()).
        """
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")

        scores = self.decision_function(X)

        # Threshold at zero, then map {0,1} -> original label names.
        idx = (scores >= 0).astype(int)
        return self.classes_[idx]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute accuracy on a labeled dataset.

        Parameters
        ----------
        X : array_like
            Feature matrix.
        y : array_like
            True labels.

        Returns
        -------
        acc : float
            Fraction of correct predictions.
        """
        y_true = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_true == y_pred))
