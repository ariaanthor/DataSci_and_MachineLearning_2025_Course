"""Unit tests for perceptron module."""

import numpy as np
import pytest

from rice_ml.supervised_learning.perceptron import Perceptron


class TestPerceptron:
    """Tests for Perceptron class."""

    def test_linearly_separable(self):
        """Test on linearly separable data."""
        # Well-separated two-class data
        X = np.array([[0, 0], [0, 1], [1, 0], [3, 3], [3, 4], [4, 3]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = Perceptron(max_iter=200, random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        # Should achieve perfect classification on well-separated data
        assert np.array_equal(preds, y)

    def test_or_gate(self):
        """Test OR gate classification."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 1])
        
        clf = Perceptron(max_iter=100, random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        assert np.array_equal(preds, y)

    def test_predict_returns_original_labels(self):
        """Test that predict returns original class labels."""
        X = np.array([[2, 1], [3, 2], [1, 4], [2, 5]], dtype=float)
        y = np.array([1, 1, -1, -1])
        
        clf = Perceptron(max_iter=100, random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        assert set(preds).issubset({1, -1})

    def test_decision_function(self):
        """Test decision function output."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        
        clf = Perceptron(max_iter=100, random_state=42).fit(X, y)
        scores = clf.decision_function(X)
        
        assert scores.shape == (4,)
        # Positive scores should correspond to class 1
        preds = clf.predict(X)
        assert all((scores >= 0) == (preds == 1))

    def test_score(self):
        """Test accuracy score."""
        # Well-separated data
        X = np.array([[0, 0], [0, 1], [1, 0], [3, 3], [3, 4], [4, 3]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = Perceptron(max_iter=200, random_state=42).fit(X, y)
        score = clf.score(X, y)
        
        assert score == 1.0

    def test_errors_tracking(self):
        """Test that errors are tracked during training."""
        # Well-separated data
        X = np.array([[0, 0], [0, 1], [1, 0], [3, 3], [3, 4], [4, 3]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = Perceptron(max_iter=200, random_state=42).fit(X, y)
        
        assert len(clf.errors_) > 0
        # Errors should decrease over time for separable data
        assert clf.errors_[-1] == 0

    def test_learning_rate(self):
        """Test effect of learning rate."""
        # Well-separated data
        X = np.array([[0, 0], [0, 1], [1, 0], [3, 3], [3, 4], [4, 3]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf_fast = Perceptron(learning_rate=10.0, max_iter=200, random_state=42).fit(X, y)
        clf_slow = Perceptron(learning_rate=0.1, max_iter=200, random_state=42).fit(X, y)
        
        # Both should eventually converge
        assert clf_fast.errors_[-1] == 0
        assert clf_slow.errors_[-1] == 0

    def test_no_intercept(self):
        """Test fitting without intercept."""
        X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]], dtype=float)
        y = np.array([1, 1, 0, 0])
        
        clf = Perceptron(fit_intercept=False, max_iter=100, random_state=42).fit(X, y)
        
        assert clf.intercept_ == 0.0

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        clf = Perceptron()
        with pytest.raises(RuntimeError):
            clf.predict([[1, 2]])

    def test_binary_only(self):
        """Test that multiclass raises error."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        y = np.array([0, 1, 2])
        
        clf = Perceptron()
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_shape_mismatch(self):
        """Test shape mismatch error."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([0, 1, 2])
        
        clf = Perceptron()
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_invalid_learning_rate(self):
        """Test invalid learning rate."""
        with pytest.raises(ValueError):
            Perceptron(learning_rate=-1.0)

    def test_convergence_tolerance(self):
        """Test early stopping with tolerance."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        
        clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42).fit(X, y)
        
        # Should converge before max_iter
        assert clf.n_iter_ < 1000

    def test_random_state_reproducibility(self):
        """Test that random state makes results reproducible."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        
        clf1 = Perceptron(max_iter=50, random_state=42).fit(X, y)
        clf2 = Perceptron(max_iter=50, random_state=42).fit(X, y)
        
        assert np.array_equal(clf1.coef_, clf2.coef_)
        assert clf1.intercept_ == clf2.intercept_