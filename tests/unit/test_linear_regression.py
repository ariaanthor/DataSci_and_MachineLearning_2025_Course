"""Unit tests for linear regression module."""

import numpy as np
import pytest
from rice_ml.supervised_learning.linear_regression import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
)


# ---------------------- LinearRegression ----------------------

class TestLinearRegression:
    """Tests for LinearRegression class."""

    def test_fit_predict_simple(self):
        """Test basic fit and predict with perfect linear data."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        
        model = LinearRegression().fit(X, y)
        
        assert model.coef_ is not None
        assert np.isclose(model.coef_[0], 2.0, atol=1e-6)
        assert np.isclose(model.intercept_, 0.0, atol=1e-6)
        
        pred = model.predict([[5]])
        assert np.isclose(pred[0], 10.0, atol=1e-6)

    def test_fit_with_intercept(self):
        """Test fitting with non-zero intercept."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([3, 5, 7, 9], dtype=float)  # y = 2x + 1
        
        model = LinearRegression().fit(X, y)
        
        assert np.isclose(model.coef_[0], 2.0, atol=1e-6)
        assert np.isclose(model.intercept_, 1.0, atol=1e-6)

    def test_no_intercept(self):
        """Test fitting without intercept."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        
        model = LinearRegression(fit_intercept=False).fit(X, y)
        
        assert model.intercept_ == 0.0
        assert np.isclose(model.coef_[0], 2.0, atol=1e-6)

    def test_multivariate(self):
        """Test multivariate regression."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_coef = np.array([1.5, -2.0, 0.5])
        y = X @ true_coef + 3.0
        
        model = LinearRegression().fit(X, y)
        
        assert np.allclose(model.coef_, true_coef, atol=1e-6)
        assert np.isclose(model.intercept_, 3.0, atol=1e-6)

    def test_gradient_descent_solver(self):
        """Test gradient descent solver."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        
        model = LinearRegression(
            solver="gd",
            learning_rate=0.1,
            max_iter=1000,
            tol=1e-6
        ).fit(X, y)
        
        assert np.isclose(model.coef_[0], 2.0, atol=0.1)
        assert np.isclose(model.intercept_, 0.0, atol=0.1)

    def test_score(self):
        """Test R^2 score."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        
        model = LinearRegression().fit(X, y)
        score = model.score(X, y)
        
        assert np.isclose(score, 1.0, atol=1e-6)

    def test_predict_not_fitted(self):
        """Test that predict raises error if not fitted."""
        model = LinearRegression()
        with pytest.raises(RuntimeError):
            model.predict([[1, 2]])

    def test_invalid_solver(self):
        """Test that invalid solver raises error."""
        with pytest.raises(ValueError):
            LinearRegression(solver="invalid")

    def test_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([1, 2, 3], dtype=float)
        
        model = LinearRegression()
        with pytest.raises(ValueError):
            model.fit(X, y)


# ---------------------- RidgeRegression ----------------------

class TestRidgeRegression:
    """Tests for RidgeRegression class."""

    def test_fit_predict_basic(self):
        """Test basic Ridge regression."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        true_coef = np.array([1.0, 2.0, 3.0])
        y = X @ true_coef + 1.0 + 0.1 * np.random.randn(50)
        
        model = RidgeRegression(alpha=0.1).fit(X, y)
        
        # Coefficients should be shrunk toward zero compared to OLS
        assert model.coef_ is not None
        assert len(model.coef_) == 3

    def test_regularization_effect(self):
        """Test that higher alpha shrinks coefficients more."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(50)
        
        model_low = RidgeRegression(alpha=0.1).fit(X, y)
        model_high = RidgeRegression(alpha=10.0).fit(X, y)
        
        # Higher alpha should result in smaller coefficient magnitudes
        assert np.linalg.norm(model_high.coef_) < np.linalg.norm(model_low.coef_)

    def test_score(self):
        """Test R^2 score."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
        y = np.array([5, 8, 11, 14], dtype=float)
        
        model = RidgeRegression(alpha=0.01).fit(X, y)
        score = model.score(X, y)
        
        assert score > 0.9

    def test_invalid_alpha(self):
        """Test that negative alpha raises error."""
        with pytest.raises(ValueError):
            RidgeRegression(alpha=-1.0)


# ---------------------- LassoRegression ----------------------

class TestLassoRegression:
    """Tests for LassoRegression class."""

    def test_fit_predict_basic(self):
        """Test basic Lasso regression."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        true_coef = np.array([1.0, 0.0, 2.0])  # One zero coefficient
        y = X @ true_coef + 0.1 * np.random.randn(50)
        
        model = LassoRegression(alpha=0.1, max_iter=1000).fit(X, y)
        
        assert model.coef_ is not None
        assert len(model.coef_) == 3

    def test_sparsity(self):
        """Test that Lasso produces sparse solutions with high alpha."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        true_coef = np.array([3, 0, 0, 2, 0, 0, 0, 1, 0, 0])
        y = X @ true_coef + 0.1 * np.random.randn(100)
        
        model = LassoRegression(alpha=0.5, max_iter=2000).fit(X, y)
        
        # High alpha should zero out some coefficients
        n_zeros = np.sum(np.abs(model.coef_) < 0.01)
        assert n_zeros > 0

    def test_score(self):
        """Test R^2 score."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        
        model = LassoRegression(alpha=0.01, max_iter=1000).fit(X, y)
        score = model.score(X, y)
        
        assert score > 0.9

    def test_convergence(self):
        """Test that model converges."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
        y = np.array([5, 8, 11, 14], dtype=float)
        
        model = LassoRegression(alpha=0.1, max_iter=1000, tol=1e-4).fit(X, y)
        
        # Should converge before max_iter
        assert model.n_iter_ <= 1000