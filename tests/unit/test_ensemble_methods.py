"""Unit tests for ensemble methods module."""

import numpy as np
import pytest

from rice_ml.supervised_learning.ensemble_methods import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)


class TestRandomForestClassifier:
    """Tests for RandomForestClassifier class."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [3, 3],
        ])
        y = np.array([0] * 30 + [1] * 30)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        assert preds.shape == (60,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self):
        """Test probability predictions."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [3, 3],
        ])
        y = np.array([0] * 20 + [1] * 20)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        proba = clf.predict_proba(X)
        
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_score(self):
        """Test accuracy score."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [4, 4],
        ])
        y = np.array([0] * 30 + [1] * 30)
        
        clf = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
        score = clf.score(X, y)
        
        assert score > 0.8

    def test_multiclass(self):
        """Test multiclass classification."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [4, 0],
            np.random.randn(20, 2) + [2, 4],
        ])
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        assert set(preds).issubset({0, 1, 2})

    def test_max_features(self):
        """Test different max_features options."""
        X = np.random.randn(50, 10)
        y = (X[:, 0] > 0).astype(int)
        
        for max_features in ["sqrt", "log2", 5, 0.5]:
            clf = RandomForestClassifier(
                n_estimators=5,
                max_features=max_features,
                random_state=42
            ).fit(X, y)
            
            preds = clf.predict(X)
            assert preds.shape == (50,)

    def test_max_depth(self):
        """Test max_depth parameter."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        
        clf = RandomForestClassifier(
            n_estimators=5,
            max_depth=3,
            random_state=42
        ).fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (50,)

    def test_no_bootstrap(self):
        """Test without bootstrap sampling."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        
        clf = RandomForestClassifier(
            n_estimators=5,
            bootstrap=False,
            random_state=42
        ).fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (50,)

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        clf = RandomForestClassifier()
        with pytest.raises(RuntimeError):
            clf.predict([[1, 2]])

    def test_invalid_n_estimators(self):
        """Test invalid n_estimators."""
        with pytest.raises(ValueError):
            RandomForestClassifier(n_estimators=0)


class TestAdaBoostClassifier:
    """Tests for AdaBoostClassifier class."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [3, 3],
        ])
        y = np.array([0] * 30 + [1] * 30)
        
        clf = AdaBoostClassifier(n_estimators=20, random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        assert preds.shape == (60,)
        assert set(preds).issubset({0, 1})

    def test_score(self):
        """Test accuracy score."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [4, 4],
        ])
        y = np.array([0] * 30 + [1] * 30)
        
        clf = AdaBoostClassifier(n_estimators=30, random_state=42).fit(X, y)
        score = clf.score(X, y)
        
        assert score > 0.7

    def test_predict_proba(self):
        """Test probability predictions."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [3, 3],
        ])
        y = np.array([0] * 20 + [1] * 20)
        
        clf = AdaBoostClassifier(n_estimators=20, random_state=42).fit(X, y)
        proba = clf.predict_proba(X)
        
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_estimator_weights(self):
        """Test that estimator weights are computed."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        
        clf = AdaBoostClassifier(n_estimators=10, random_state=42).fit(X, y)
        
        assert clf.estimator_weights_ is not None
        assert len(clf.estimator_weights_) == 10
        assert np.all(clf.estimator_weights_ > 0)

    def test_learning_rate(self):
        """Test learning rate effect."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        
        clf_fast = AdaBoostClassifier(
            n_estimators=10, learning_rate=1.0, random_state=42
        ).fit(X, y)
        
        clf_slow = AdaBoostClassifier(
            n_estimators=10, learning_rate=0.1, random_state=42
        ).fit(X, y)
        
        # Both should produce valid predictions
        assert clf_fast.predict(X).shape == (50,)
        assert clf_slow.predict(X).shape == (50,)

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        clf = AdaBoostClassifier()
        with pytest.raises(RuntimeError):
            clf.predict([[1, 2]])

    def test_invalid_n_estimators(self):
        """Test invalid n_estimators."""
        with pytest.raises(ValueError):
            AdaBoostClassifier(n_estimators=0)

    def test_invalid_learning_rate(self):
        """Test invalid learning rate."""
        with pytest.raises(ValueError):
            AdaBoostClassifier(learning_rate=-1.0)


class TestBaggingClassifier:
    """Tests for BaggingClassifier class."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [3, 3],
        ])
        y = np.array([0] * 30 + [1] * 30)
        
        clf = BaggingClassifier(n_estimators=10, random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        assert preds.shape == (60,)
        assert set(preds).issubset({0, 1})

    def test_score(self):
        """Test accuracy score."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [4, 4],
        ])
        y = np.array([0] * 30 + [1] * 30)
        
        clf = BaggingClassifier(n_estimators=10, random_state=42).fit(X, y)
        score = clf.score(X, y)
        
        assert score > 0.7

    def test_max_samples(self):
        """Test max_samples parameter."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        
        clf = BaggingClassifier(
            n_estimators=5,
            max_samples=0.5,
            random_state=42
        ).fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (50,)

    def test_max_features(self):
        """Test max_features parameter."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = (X[:, 0] > 0).astype(int)
        
        clf = BaggingClassifier(
            n_estimators=5,
            max_features=0.6,
            random_state=42
        ).fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (50,)

    def test_no_bootstrap(self):
        """Test without bootstrap."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        
        clf = BaggingClassifier(
            n_estimators=5,
            bootstrap=False,
            random_state=42
        ).fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (50,)

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        clf = BaggingClassifier()
        with pytest.raises(RuntimeError):
            clf.predict([[1, 2]])

    def test_invalid_parameters(self):
        """Test invalid parameters."""
        with pytest.raises(ValueError):
            BaggingClassifier(n_estimators=0)
        
        with pytest.raises(ValueError):
            BaggingClassifier(max_samples=1.5)
        
        with pytest.raises(ValueError):
            BaggingClassifier(max_features=0)