# Linear Regression

This module implements common linear regression variants **from scratch**.

---

## What’s Included

### `LinearRegression`
Ordinary least squares (OLS) with two solvers:
- `solver="closed"`: closed-form solution using a pseudo-inverse
- `solver="gd"`: gradient descent on mean squared error

Options:
- `fit_intercept` (learn a bias term)
- `learning_rate`, `max_iter`, `tol` (for gradient descent)

### `RidgeRegression`
OLS with **L2 regularization** (ridge):
- Minimizes squared error with an `alpha * ||w||^2` penalty
- Uses a closed-form linear solve
- Centers data when `fit_intercept=True` and recovers the intercept afterward

### `LassoRegression`
OLS with **L1 regularization** (lasso):
- Uses **coordinate descent** with a soft-thresholding update
- Supports `max_iter` and `tol` for convergence control
- Centers data when `fit_intercept=True` and recovers the intercept afterward

---

## Quick Start

```python
import numpy as np
from rice_ml.supervised_learning.linear_regression import (
    LinearRegression, RidgeRegression, LassoRegression
)

X = np.array([[0.], [1.], [2.], [3.]], dtype=float)
y = np.array([1., 3., 5., 7.], dtype=float)  # y = 2x + 1

ols = LinearRegression(solver="closed").fit(X, y)
print(ols.coef_, ols.intercept_)
print(ols.predict([[4.]]))

ridge = RidgeRegression(alpha=1.0).fit(X, y)
print(ridge.predict([[4.]]))

lasso = LassoRegression(alpha=0.1).fit(X, y)
print(lasso.predict([[4.]]))
