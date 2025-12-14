"""
linear_regression.py

This file implements three classic linear models:

- LinearRegression:
    Ordinary least squares (OLS). Two solvers are provided:
      * "closed" : solves using a pseudo-inverse normal equation
      * "gd"     : gradient descent on mean squared error

- RidgeRegression:
    OLS with L2 shrinkage (ridge). Uses a closed-form solve.

- LassoRegression:
    OLS with L1 penalty (lasso). Uses coordinate descent with soft-thresholding.

The implementations favor readability and explicit steps over speed.

Mini example
------------
>>> import numpy as np
>>> from rice_ml.supervised_learning.linear_regression import RidgeRegression
>>> X = np.array([[0., 0.],
...               [1., 0.],
...               [0., 1.],
...               [1., 1.]])
>>> y = np.array([0., 1., 1., 2.])
>>> m = RidgeRegression(alpha=0.1).fit(X, y)
>>> float(np.round(m.predict([[2., 2.]])[0], 6)) > 0.0
True
"""

from __future__ import annotations
from typing import Literal, Optional, Union, Sequence

import numpy as np

__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ==============================
# Input checks / normalization
# ==============================

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Coerce an array-like into a 2D float NumPy array.

    Behavior
    --------
    - If X is 1D, it is treated as a single feature and reshaped to (n, 1).
    - Only 1D/2D inputs are accepted.
    - Empty inputs are rejected.
    - Non-numeric inputs are converted to float when possible.

    Parameters
    ----------
    X : ArrayLike
        Feature data.
    name : str
        Name used for readable error messages.

    Returns
    -------
    np.ndarray
        Float matrix with shape (n_samples, n_features).
    """
    arr = np.asarray(X)

    # Convenience: allow a single feature vector without forcing callers to reshape
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got {arr.ndim}D.")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    # Force numeric representation (float) for all solvers
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)

    return arr


def _ensure_1d_float(y: ArrayLike, name: str = "y") -> np.ndarray:
    """
    Coerce y into a 1D float vector.

    Parameters
    ----------
    y : ArrayLike
        Target values (regression).
    name : str
        Name used for readable error messages.

    Returns
    -------
    np.ndarray
        Float vector with shape (n_samples,).

    Raises
    ------
    ValueError
        If `y` is not 1D or is empty.
    TypeError
        If conversion to float fails.
    """
    arr = np.asarray(y)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
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


# ==============================
# Ordinary Least Squares
# ==============================

class LinearRegression:
    """
    Ordinary least squares (OLS) linear regression.

    The model predicts:
        y_hat = X @ coef_ + intercept_

    Two fitting approaches are available:
    - solver="closed": pseudo-inverse solution (stable for rank-deficient X)
    - solver="gd": gradient descent on MSE

    Parameters
    ----------
    fit_intercept : bool, default=True
        If True, learn an intercept term.
    solver : {"closed", "gd"}, default="closed"
        Optimization strategy.
    learning_rate : float, default=0.01
        Step size for gradient descent (only used when solver="gd").
    max_iter : int, default=1000
        Iteration cap for gradient descent.
    tol : float, default=1e-6
        Stops early when the gradient norm is below this threshold.

    Attributes
    ----------
    coef_ : np.ndarray | None
        Weight vector of shape (n_features,). Set after fitting.
    intercept_ : float
        Bias term. 0.0 when `fit_intercept=False`.
    n_iter_ : int
        Number of iterations executed (1 for closed-form).

    Example
    -------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.linear_regression import LinearRegression
    >>> X = np.array([[0.], [1.], [2.]], dtype=float)
    >>> y = np.array([1., 3., 5.], dtype=float)  # y = 2x + 1
    >>> lr = LinearRegression(solver="closed").fit(X, y)
    >>> (round(float(lr.intercept_), 6), round(float(lr.coef_[0]), 6))
    (1.0, 2.0)
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        solver: Literal["closed", "gd"] = "closed",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        # Parameter checks kept simple and explicit
        if solver not in ("closed", "gd"):
            raise ValueError("solver must be 'closed' or 'gd'.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """
        Fit coefficients (and optionally an intercept) to training data.

        Parameters
        ----------
        X : ArrayLike
            Training features, shape (n_samples, n_features).
        y : ArrayLike
            Training targets, shape (n_samples,).

        Returns
        -------
        self : LinearRegression
            The fitted estimator.

        Raises
        ------
        ValueError
            If X and y have incompatible sample counts.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_float(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples; "
                f"got {X_arr.shape[0]} vs {y_arr.shape[0]}."
            )

        # Dispatch to the selected solver
        if self.solver == "closed":
            self._fit_closed(X_arr, y_arr)
        else:
            self._fit_gd(X_arr, y_arr)

        return self

    def _fit_closed(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Closed-form fit using a pseudo-inverse solve.

        Uses:
            theta = pinv(X_b^T X_b) X_b^T y
        where X_b includes a bias column if fit_intercept=True.
        """
        if self.fit_intercept:
            X_b = np.c_[np.ones(X.shape[0]), X]
        else:
            X_b = X

        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

        # With the normal equation, we conceptually "iterate" once
        self.n_iter_ = 1

    def _fit_gd(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Gradient descent on mean squared error.

        Updates:
            coef  <- coef  - lr * (1/n) X^T (X coef + b - y)
            b     <- b     - lr * (1/n) sum(X coef + b - y)   (if fit_intercept)
        """
        n_samples, n_features = X.shape

        # Start from zeros (simple baseline initialization)
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0  # same value whether intercept is used or not

        for i in range(self.max_iter):
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y

            grad_coef = (1 / n_samples) * (X.T @ error)
            grad_intercept = (1 / n_samples) * np.sum(error) if self.fit_intercept else 0.0

            self.coef_ -= self.learning_rate * grad_coef
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_intercept

            # Stop when the update direction becomes tiny
            if np.linalg.norm(grad_coef) < self.tol:
                self.n_iter_ = i + 1
                return

        self.n_iter_ = self.max_iter

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Generate predictions for input features.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_arr = _ensure_2d_float(X, "X")

        if X_arr.shape[1] != len(self.coef_):
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {len(self.coef_)}.")

        return X_arr @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        R^2 score (coefficient of determination).

        Returns 1.0 for perfect fit and can return 0.0 when the baseline
        model (predicting the mean) matches the performance.

        Note: if y is constant, this method mirrors the original behavior:
        return 1.0 if predictions are perfect, otherwise 0.0.
        """
        y_arr = _ensure_1d_float(y, "y")
        y_pred = self.predict(X)

        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return float(1.0 - ss_res / ss_tot)


# ==============================
# Ridge (L2) Regression
# ==============================

class RidgeRegression:
    """
    Linear regression with L2 regularization (ridge).

    Objective (informal):
        minimize ||y - Xw||^2 + alpha * ||w||^2

    Fitting is done in closed form on centered data when `fit_intercept=True`.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength (>= 0).
    fit_intercept : bool, default=True
        If True, center X and y and recover the intercept.

    Attributes
    ----------
    coef_ : np.ndarray | None
        Learned weights.
    intercept_ : float
        Bias term.

    Example
    -------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.linear_regression import RidgeRegression
    >>> X = np.array([[1., 0.], [0., 1.], [1., 1.]])
    >>> y = np.array([1., 1., 2.])
    >>> rr = RidgeRegression(alpha=1.0).fit(X, y)
    >>> rr.predict([[2., 2.]]).shape
    (1,)
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RidgeRegression":
        """
        Fit ridge regression parameters via a linear solve.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_float(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        n_samples, n_features = X_arr.shape

        # Centering makes intercept handling straightforward
        if self.fit_intercept:
            X_mean = X_arr.mean(axis=0)
            y_mean = y_arr.mean()
            X_centered = X_arr - X_mean
            y_centered = y_arr - y_mean
        else:
            X_centered = X_arr
            y_centered = y_arr
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        # (X^T X + alpha I) w = X^T y
        A = X_centered.T @ X_centered + self.alpha * np.eye(n_features)
        b = X_centered.T @ y_centered
        self.coef_ = np.linalg.solve(A, b)

        # Un-center to recover intercept
        self.intercept_ = float(y_mean - X_mean @ self.coef_) if self.fit_intercept else 0.0
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict targets with the ridge model.
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != len(self.coef_):
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {len(self.coef_)}.")
        return X_arr @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        R^2 score (same definition as LinearRegression.score).
        """
        y_arr = _ensure_1d_float(y, "y")
        y_pred = self.predict(X)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)


# ==============================
# Lasso (L1) Regression
# ==============================

class LassoRegression:
    """
    Linear regression with L1 regularization (lasso).

    This implementation uses coordinate descent with the soft-threshold operator.

    Minimizes (as implemented):
        ||y - Xw||^2 / (2n) + alpha * ||w||_1

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization amount (>= 0).
    fit_intercept : bool, default=True
        Center data and recover intercept.
    max_iter : int, default=1000
        Max coordinate-descent passes.
    tol : float, default=1e-4
        Stops when the largest coefficient change is below `tol`.

    Attributes
    ----------
    coef_ : np.ndarray | None
        Learned coefficients.
    intercept_ : float
        Bias term.
    n_iter_ : int
        Completed iterations (passes over all coordinates).

    Example
    -------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.linear_regression import LassoRegression
    >>> X = np.array([[0.], [1.], [2.], [3.]], dtype=float)
    >>> y = np.array([0., 1., 2., 3.], dtype=float)
    >>> ls = LassoRegression(alpha=0.0).fit(X, y)
    >>> float(np.round(ls.predict([[4.]])[0], 6))
    4.0
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def _soft_threshold(self, x: float, lam: float) -> float:
        """
        Soft-thresholding primitive used by coordinate descent.

        This is the proximal operator for the L1 norm.
        """
        if x > lam:
            return x - lam
        elif x < -lam:
            return x + lam
        else:
            return 0.0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LassoRegression":
        """
        Fit lasso parameters by coordinate descent.

        The algorithm iteratively updates each coefficient w_j while holding
        the others fixed, updating a maintained residual vector to keep the
        update per-coordinate efficient.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_float(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        n_samples, n_features = X_arr.shape

        # Center inputs to isolate intercept
        if self.fit_intercept:
            X_mean = X_arr.mean(axis=0)
            y_mean = y_arr.mean()
            X_c = X_arr - X_mean
            y_c = y_arr - y_mean
        else:
            X_c = X_arr
            y_c = y_arr
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        # Useful precomputations: column squared norms
        X_col_norms_sq = np.sum(X_c ** 2, axis=0)

        # Start with zero weights; residual starts as y (since predictions are 0)
        self.coef_ = np.zeros(n_features)
        residual = y_c.copy()

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                # Skip features with no variance
                if X_col_norms_sq[j] == 0:
                    continue

                # "Undo" current contribution of feature j
                residual += X_c[:, j] * self.coef_[j]

                # Correlation between column and current residual
                rho_j = X_c[:, j] @ residual

                # Coordinate update (same as original logic)
                self.coef_[j] = self._soft_threshold(
                    rho_j / n_samples,
                    self.alpha
                ) / (X_col_norms_sq[j] / n_samples)

                # Apply updated contribution back into residual
                residual -= X_c[:, j] * self.coef_[j]

            # Convergence: monitor maximum absolute coefficient change
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        # Recover intercept in the original coordinate system
        self.intercept_ = float(y_mean - X_mean @ self.coef_) if self.fit_intercept else 0.0
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict with the lasso model.
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != len(self.coef_):
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {len(self.coef_)}.")
        return X_arr @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        R^2 score (same definition as LinearRegression.score).
        """
        y_arr = _ensure_1d_float(y, "y")
        y_pred = self.predict(X)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)
