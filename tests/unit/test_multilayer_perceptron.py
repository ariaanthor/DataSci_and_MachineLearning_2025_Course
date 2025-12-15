"""Unit tests for multilayer perceptron module."""

import numpy as np
import pytest

from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier, MLPRegressor


class TestMLPClassifier:
    """Tests for MLPClassifier class."""

    def test_xor_problem(self):
        """Test XOR problem (not linearly separable)."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        
        clf = MLPClassifier(
            hidden_layer_sizes=(8,),
            max_iter=2000,
            random_state=1,
            learning_rate=0.1
        ).fit(X, y)
        
        preds = clf.predict(X)
        # Should solve XOR perfectly or nearly
        assert np.mean(preds == y) >= 0.75

    def test_binary_classification(self):
        """Test basic binary classification."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [3, 3],
        ])
        y = np.array([0] * 30 + [1] * 30)
        
        clf = MLPClassifier(
            hidden_layer_sizes=(10,),
            max_iter=500,
            random_state=42
        ).fit(X, y)
        
        score = clf.score(X, y)
        assert score > 0.8

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [4, 0],
            np.random.randn(20, 2) + [2, 4],
        ])
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        clf = MLPClassifier(
            hidden_layer_sizes=(20,),
            max_iter=500,
            random_state=42
        ).fit(X, y)
        
        assert len(clf.classes_) == 3
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba(self):
        """Test probability predictions."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [3, 3],
        ])
        y = np.array([0] * 20 + [1] * 20)
        
        clf = MLPClassifier(
            hidden_layer_sizes=(10,),
            max_iter=300,
            random_state=42
        ).fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_different_activations(self):
        """Test different activation functions."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        
        for activation in ['relu', 'sigmoid', 'tanh']:
            clf = MLPClassifier(
                hidden_layer_sizes=(8,),
                activation=activation,
                max_iter=1000,
                random_state=42
            ).fit(X, y)
            
            # Should produce valid predictions
            preds = clf.predict(X)
            assert set(preds).issubset({0, 1})

    def test_loss_curve(self):
        """Test that loss curve is tracked."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        
        clf = MLPClassifier(
            hidden_layer_sizes=(8,),
            max_iter=100,
            random_state=42
        ).fit(X, y)
        
        assert len(clf.loss_curve_) > 0
        # Loss should generally decrease
        assert clf.loss_curve_[-1] <= clf.loss_curve_[0]

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        clf = MLPClassifier()
        with pytest.raises(RuntimeError):
            clf.predict([[1, 2]])

    def test_invalid_hidden_layers(self):
        """Test invalid hidden layer sizes."""
        with pytest.raises(ValueError):
            MLPClassifier(hidden_layer_sizes=())

    def test_invalid_activation(self):
        """Test invalid activation function."""
        with pytest.raises(ValueError):
            MLPClassifier(activation="invalid")

    def test_single_class_error(self):
        """Test error with single class."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([0, 0])
        
        clf = MLPClassifier()
        with pytest.raises(ValueError):
            clf.fit(X, y)


class TestMLPRegressor:
    """Tests for MLPRegressor class."""

    def test_simple_regression(self):
        """Test simple linear regression."""
        np.random.seed(42)
        # Use standardized data for better numerical stability
        X = np.random.randn(50, 1)
        y = 0.5 * X.ravel() + 0.1 * np.random.randn(50)
        
        reg = MLPRegressor(
            hidden_layer_sizes=(10,),
            max_iter=500,
            random_state=42,
            learning_rate=0.001
        ).fit(X, y)
        
        preds = reg.predict(X)
        # Just check predictions are reasonable (finite and close to y range)
        assert np.all(np.isfinite(preds))
        assert np.abs(preds.mean() - y.mean()) < 1.0

    def test_nonlinear_regression(self):
        """Test nonlinear regression."""
        np.random.seed(42)
        X = np.linspace(-3, 3, 100).reshape(-1, 1)
        y = np.sin(X.ravel()) + 0.1 * np.random.randn(100)
        
        reg = MLPRegressor(
            hidden_layer_sizes=(20, 20),
            max_iter=1000,
            random_state=42
        ).fit(X, y)
        
        preds = reg.predict(X)
        assert preds.shape == y.shape

    def test_multivariate_input(self):
        """Test multivariate regression."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
        
        reg = MLPRegressor(
            hidden_layer_sizes=(10,),
            max_iter=500,
            random_state=42
        ).fit(X, y)
        
        preds = reg.predict(X)
        assert preds.shape == (100,)

    def test_loss_decreases(self):
        """Test that loss decreases during training."""
        np.random.seed(42)
        X = np.linspace(0, 5, 30).reshape(-1, 1)
        y = X.ravel() ** 2
        
        reg = MLPRegressor(
            hidden_layer_sizes=(10,),
            max_iter=200,
            random_state=42
        ).fit(X, y)
        
        assert len(reg.loss_curve_) > 0
        # Loss should decrease
        assert reg.loss_curve_[-1] < reg.loss_curve_[0]

    def test_score(self):
        """Test R^2 score."""
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        
        reg = MLPRegressor(
            hidden_layer_sizes=(10,),
            max_iter=1000,
            random_state=42
        ).fit(X, y)
        
        score = reg.score(X, y)
        assert score > 0.5

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        reg = MLPRegressor()
        with pytest.raises(RuntimeError):
            reg.predict([[1, 2]])

    def test_shape_mismatch(self):
        """Test shape mismatch error."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([1, 2, 3])
        
        reg = MLPRegressor()
        with pytest.raises(ValueError):
            reg.fit(X, y)