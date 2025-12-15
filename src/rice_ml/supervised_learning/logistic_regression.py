"""
logistic_regression.py

This module contains a gradient-descent optimizer for logistic regression
that can handle:
- binary classification (sigmoid)
- multiclass classification via:
    * "multinomial" softmax regression
    * "ovr" (one-vs-rest)

Regularization
--------------
- penalty="none": no weight penalty
- penalty="l2": adds an L2 penalty scaled by 1/C

The public API mirrors common ML libraries: fit / predict / predict_proba / score.

Quick demo
----------
>>> import numpy as np
>>> from rice_ml.supervised_learning.logistic_regression import LogisticRegression
>>> X = np.array([[0., 0.],
...               [0., 1.],
...               [1., 0.],
...               [1., 1.]])
>>> y = np.array([0, 1, 1, 1])
>>> clf = LogisticRegression(learning_rate=0.2, max_iter=2000, tol=1e-6).fit(X, y)
>>> clf.predict([[0., 0.], [1., 1.]]).tolist()
[0, 1]
"""

from __future__ import annotations
from typing import Literal, Optional, Union, Sequence

import numpy as np

__all__ = ["LogisticRegression"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ==============================
# Helpers: shape + dtype checks
# ==============================

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert X into a 2D float array.

    Notes
    -----
    - A 1D input is interpreted as a single feature and reshaped to (n, 1).
    - Raises on empty inputs.
    - Attempts float conversion for non-numeric dtypes.
    """
    arr = np.asarray(X)

    # Allow "single feature" input as 1D
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    # Force float type to simplify downstream math
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
    Convert y into a 1D array.

    Labels may be any dtype (ints, strings, etc.) as long as there are >=2 classes.
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


# ==============================
# Activations (stable versions)
# ==============================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation with overflow protection.

    Clipping keeps exp() from blowing up for large magnitude inputs.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax activation computed row-wise.

    Uses z - max(z) trick to reduce overflow/underflow.
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class LogisticRegression:
    """
    Logistic regression classifier (binary + multiclass).

    This implementation trains parameters via (batch) gradient descent.

    Multiclass behavior
    -------------------
    multi_class="auto":
        - if there are exactly 2 classes -> binary sigmoid model
        - otherwise -> multinomial softmax model
    multi_class="multinomial":
        - softmax regression (single model for all classes)
    multi_class="ovr":
        - one-vs-rest (train one binary classifier per class)

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size used during optimization.
    max_iter : int, default=1000
        Maximum number of gradient steps.
    tol : float, default=1e-4
        Convergence threshold (checked on parameter change).
    fit_intercept : bool, default=True
        If True, a bias term is learned.
    penalty : {"none", "l2"}, default="none"
        L2 penalty adds (1/C) * w to the gradient.
    C : float, default=1.0
        Inverse regularization factor (bigger C -> weaker regularization).
    multi_class : {"auto", "ovr", "multinomial"}, default="auto"
        Strategy selection for multiclass data.

    Attributes
    ----------
    classes_ : np.ndarray | None
        Sorted unique labels observed during fit.
    coef_ : np.ndarray | None
        Learned weights:
        - binary: shape (1, n_features)
        - multinomial/ovr: shape (n_classes, n_features)
    intercept_ : np.ndarray | None
        Learned biases:
        - binary: shape (1,)
        - multinomial/ovr: shape (n_classes,)
    n_iter_ : int
        Iterations performed (binary/multinomial may stop early).

    Example
    -------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.logistic_regression import LogisticRegression
    >>> X = np.array([[1., 0.],
    ...               [0., 1.],
    ...               [1., 1.],
    ...               [2., 1.]])
    >>> y = np.array(["neg", "neg", "pos", "pos"], dtype=object)
    >>> m = LogisticRegression(learning_rate=0.1, max_iter=1500).fit(X, y)
    >>> m.predict([[0., 0.], [2., 2.]]).tolist()
    ['neg', 'pos']
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        penalty: Literal["none", "l2"] = "none",
        C: float = 1.0,
        multi_class: Literal["auto", "ovr", "multinomial"] = "auto",
    ) -> None:
        # Validate configuration
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if C <= 0:
            raise ValueError("C must be positive.")
        if penalty not in ("none", "l2"):
            raise ValueError("penalty must be 'none' or 'l2'.")
        if multi_class not in ("auto", "ovr", "multinomial"):
            raise ValueError("multi_class must be 'auto', 'ovr', or 'multinomial'.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.C = C
        self.multi_class = multi_class

        # Learned parameters (set after calling fit)
        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    # -------------------------
    # Public training interface
    # -------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """
        Train the classifier on (X, y).

        Parameters
        ----------
        X : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        y : ArrayLike
            Labels of shape (n_samples,).

        Returns
        -------
        self : LogisticRegression
            Fitted estimator.

        Raises
        ------
        ValueError
            If there are fewer than 2 unique labels.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("y must contain at least 2 classes.")

        # Decide training strategy
        if self.multi_class == "auto":
            strategy = "binary" if n_classes == 2 else "multinomial"
        elif self.multi_class == "ovr":
            strategy = "ovr"
        else:
            strategy = "multinomial"

        # Binary data should use sigmoid unless explicitly forced to OvR
        if n_classes == 2 and strategy != "ovr":
            strategy = "binary"

        if strategy == "binary":
            self._fit_binary(X_arr, y_arr)
        elif strategy == "multinomial":
            self._fit_multinomial(X_arr, y_arr)
        else:
            self._fit_ovr(X_arr, y_arr)

        return self

    # -------------------------
    # Internal: binary training
    # -------------------------

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Binary logistic regression with sigmoid activation.

        Implementation detail
        ---------------------
        The "positive" class is taken to be classes_[1] (sorted order),
        and y is internally mapped to {0, 1}.
        """
        n_samples, n_features = X.shape

        # Map labels to {0, 1} where 1 corresponds to the second unique class
        y_binary = (y == self.classes_[1]).astype(float)

        w = np.zeros(n_features)
        b = 0.0

        # Same as original: L2 gradient term uses reg_strength = 1/C
        reg_strength = 1.0 / self.C if self.penalty == "l2" else 0.0

        for i in range(self.max_iter):
            z = X @ w + b
            h = _sigmoid(z)

            error = h - y_binary
            grad_w = (1 / n_samples) * (X.T @ error)
            grad_b = (1 / n_samples) * np.sum(error)

            if reg_strength > 0:
                grad_w += reg_strength * w

            w_old = w.copy()

            # Parameter update
            w -= self.learning_rate * grad_w
            if self.fit_intercept:
                b -= self.learning_rate * grad_b

            # Early stop based on coefficient movement
            if np.max(np.abs(w - w_old)) < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])

    # -----------------------------
    # Internal: multinomial training
    # -----------------------------

    def _fit_multinomial(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Multiclass softmax regression (single model, all classes).

        The weight matrix is stored as:
            self.coef_ shape (n_classes, n_features)
        to match the rest of the module’s conventions.
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Build an index mapping and one-hot targets
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[label] for label in y])

        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y_idx] = 1

        W = np.zeros((n_features, n_classes))
        b = np.zeros(n_classes)

        reg_strength = 1.0 / self.C if self.penalty == "l2" else 0.0

        for i in range(self.max_iter):
            z = X @ W + b
            probs = _softmax(z)

            error = probs - Y_onehot
            grad_W = (1 / n_samples) * (X.T @ error)
            grad_b = (1 / n_samples) * np.sum(error, axis=0)

            if reg_strength > 0:
                grad_W += reg_strength * W

            W_old = W.copy()

            W -= self.learning_rate * grad_W
            if self.fit_intercept:
                b -= self.learning_rate * grad_b

            if np.max(np.abs(W - W_old)) < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        # Store in (n_classes, n_features) layout (as in original code)
        self.coef_ = W.T
        self.intercept_ = b

    # -------------------------
    # Internal: OvR training
    # -------------------------

    def _fit_ovr(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        One-vs-Rest training loop.

        For each class k:
            treat y==class_k as positive, others as negative
            train a sigmoid model
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)

        reg_strength = 1.0 / self.C if self.penalty == "l2" else 0.0

        for idx, cls in enumerate(self.classes_):
            y_binary = (y == cls).astype(float)
            w = np.zeros(n_features)
            b = 0.0

            # NOTE: original implementation does not early-stop in OvR
            for _ in range(self.max_iter):
                z = X @ w + b
                h = _sigmoid(z)

                error = h - y_binary
                grad_w = (1 / n_samples) * (X.T @ error)
                grad_b = (1 / n_samples) * np.sum(error)

                if reg_strength > 0:
                    grad_w += reg_strength * w

                w -= self.learning_rate * grad_w
                if self.fit_intercept:
                    b -= self.learning_rate * grad_b

            self.coef_[idx] = w
            self.intercept_[idx] = b

        self.n_iter_ = self.max_iter

    # -------------------------
    # Inference utilities
    # -------------------------

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Return class probabilities for each sample.

        Binary case:
            returns two columns [P(class0), P(class1)]
        Multiclass/OvR case:
            returns softmax over class scores.

        Parameters
        ----------
        X : ArrayLike
            Input samples.

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes).
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != self.coef_.shape[1]:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.coef_.shape[1]}.")

        n_classes = len(self.classes_)

        # Binary model is stored with coef_ shape (1, n_features)
        if n_classes == 2 and self.coef_.shape[0] == 1:
            z = X_arr @ self.coef_[0] + self.intercept_[0]
            p_pos = _sigmoid(z)
            return np.column_stack([1 - p_pos, p_pos])

        # Multiclass or OvR: compute class scores and normalize
        z = X_arr @ self.coef_.T + self.intercept_
        return _softmax(z)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict labels for samples in X.
        """
        proba = self.predict_proba(X)
        best_idx = np.argmax(proba, axis=1)
        return self.classes_[best_idx]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Classification accuracy on (X, y).
        """
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))
