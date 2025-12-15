"""Unit tests for logistic regression module."""

import numpy as np
import pytest

from rice_ml.supervised_learning.logistic_regression import LogisticRegression


class TestLogisticRegression:
    """Tests for LogisticRegression class."""

    def test_binary_classification_basic(self):
        """Test basic binary classification."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = LogisticRegression(max_iter=1000, learning_rate=0.1).fit(X, y)
        
        assert clf.classes_ is not None
        assert len(clf.classes_) == 2
        assert clf.coef_ is not None

    def test_predict_proba_sums_to_one(self):
        """Test that predict_proba returns valid probabilities."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        
        clf = LogisticRegression(max_iter=500).fit(X, y)
        proba = clf.predict_proba(X)
        
        # Each row should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
        # All probabilities should be in [0, 1]
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_returns_valid_labels(self):
        """Test that predict returns valid class labels."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        
        clf = LogisticRegression(max_iter=500).fit(X, y)
        preds = clf.predict(X)
        
        assert set(preds).issubset(set(y))

    def test_score_accuracy(self):
        """Test accuracy score."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = LogisticRegression(max_iter=1000, learning_rate=0.5).fit(X, y)
        score = clf.score(X, y)
        
        assert 0 <= score <= 1

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        np.random.seed(42)
        # Create 3-class problem
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [3, 0],
            np.random.randn(20, 2) + [1.5, 3],
        ])
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        clf = LogisticRegression(max_iter=1000, learning_rate=0.1).fit(X, y)
        
        assert len(clf.classes_) == 3
        assert clf.coef_.shape[0] == 3
        
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1, 2})

    def test_multiclass_proba(self):
        """Test multiclass probability predictions."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(15, 2) + [0, 0],
            np.random.randn(15, 2) + [3, 0],
            np.random.randn(15, 2) + [1.5, 3],
        ])
        y = np.array([0] * 15 + [1] * 15 + [2] * 15)
        
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        proba = clf.predict_proba(X)
        
        assert proba.shape == (45, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_l2_regularization(self):
        """Test L2 regularization effect."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        clf_no_reg = LogisticRegression(penalty="none", max_iter=1000).fit(X, y)
        clf_l2 = LogisticRegression(penalty="l2", C=0.1, max_iter=1000).fit(X, y)
        
        # L2 should shrink coefficients
        assert np.linalg.norm(clf_l2.coef_) <= np.linalg.norm(clf_no_reg.coef_)

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        clf = LogisticRegression()
        with pytest.raises(RuntimeError):
            clf.predict([[1, 2]])

    def test_single_class_error(self):
        """Test error when only one class is present."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([0, 0])
        
        clf = LogisticRegression()
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_shape_mismatch_error(self):
        """Test error on shape mismatch."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([0, 1, 2])
        
        clf = LogisticRegression()
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_feature_mismatch_error(self):
        """Test error when predict has wrong number of features."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([0, 1])
        
        clf = LogisticRegression(max_iter=100).fit(X, y)
        with pytest.raises(ValueError):
            clf.predict([[1, 2, 3]])

    def test_ovr_strategy(self):
        """Test One-vs-Rest strategy."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(15, 2) + [0, 0],
            np.random.randn(15, 2) + [3, 0],
            np.random.randn(15, 2) + [1.5, 3],
        ])
        y = np.array([0] * 15 + [1] * 15 + [2] * 15)
        
        clf = LogisticRegression(multi_class="ovr", max_iter=1000).fit(X, y)
        
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1, 2})