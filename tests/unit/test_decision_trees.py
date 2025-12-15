"""Unit tests for decision tree classifier and regressor modules."""

import numpy as np
import pytest

from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier
from rice_ml.supervised_learning.regression_trees import DecisionTreeRegressor


class TestDecisionTreeClassifier:
    """Tests for DecisionTreeClassifier class."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        assert preds.shape == (4,)
        assert set(preds).issubset({0, 1})

    def test_perfect_split(self):
        """Test perfectly separable data."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        # Should achieve perfect accuracy
        assert np.array_equal(preds, y)

    def test_predict_proba(self):
        """Test probability predictions."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(random_state=42).fit(X, y)
        proba = clf.predict_proba(X)
        
        assert proba.shape == (4, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_max_depth(self):
        """Test max_depth parameter."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        
        clf_shallow = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
        clf_deep = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X, y)
        
        # Both should produce valid predictions
        assert clf_shallow.predict(X).shape == (100,)
        assert clf_deep.predict(X).shape == (100,)
        
        # Deep tree should generally fit better
        score_shallow = np.mean(clf_shallow.predict(X) == y)
        score_deep = np.mean(clf_deep.predict(X) == y)
        assert score_deep >= score_shallow

    def test_min_samples_split(self):
        """Test min_samples_split parameter."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        
        clf = DecisionTreeClassifier(
            min_samples_split=10,
            random_state=42
        ).fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (50,)

    def test_min_samples_leaf(self):
        """Test min_samples_leaf parameter."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        
        clf = DecisionTreeClassifier(
            min_samples_leaf=5,
            random_state=42
        ).fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (50,)

    def test_max_features(self):
        """Test max_features parameter."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = (X[:, 0] > 0).astype(int)
        
        for max_features in [5, 0.5]:
            clf = DecisionTreeClassifier(
                max_features=max_features,
                random_state=42
            ).fit(X, y)
            
            preds = clf.predict(X)
            assert preds.shape == (50,)

    def test_multiclass(self):
        """Test multiclass classification."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [3, 0],
            np.random.randn(20, 2) + [1.5, 3],
        ])
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        clf = DecisionTreeClassifier(random_state=42).fit(X, y)
        preds = clf.predict(X)
        
        assert set(preds).issubset({0, 1, 2})

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        clf = DecisionTreeClassifier()
        with pytest.raises(RuntimeError):
            clf.predict([[1, 2]])

    def test_shape_mismatch(self):
        """Test shape mismatch error."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([0, 1, 2])
        
        clf = DecisionTreeClassifier()
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_non_integer_labels(self):
        """Test error with non-integer labels."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([0.5, 1.5])
        
        clf = DecisionTreeClassifier()
        with pytest.raises(ValueError):
            clf.fit(X, y)


class TestDecisionTreeRegressor:
    """Tests for DecisionTreeRegressor class."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict."""
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        reg = DecisionTreeRegressor(random_state=42).fit(X, y)
        preds = reg.predict(X)
        
        assert preds.shape == (5,)

    def test_perfect_fit(self):
        """Test perfect fit on training data with no depth limit."""
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        reg = DecisionTreeRegressor(random_state=42).fit(X, y)
        preds = reg.predict(X)
        
        # Should fit perfectly
        assert np.allclose(preds, y)

    def test_max_depth(self):
        """Test max_depth parameter."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] ** 2 + X[:, 1]
        
        reg_shallow = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y)
        reg_deep = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X, y)
        
        # Deep tree should fit better
        score_shallow = reg_shallow.score(X, y)
        score_deep = reg_deep.score(X, y)
        assert score_deep >= score_shallow

    def test_min_samples_split(self):
        """Test min_samples_split parameter."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        
        reg = DecisionTreeRegressor(
            min_samples_split=10,
            random_state=42
        ).fit(X, y)
        
        preds = reg.predict(X)
        assert preds.shape == (50,)

    def test_min_samples_leaf(self):
        """Test min_samples_leaf parameter."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        
        reg = DecisionTreeRegressor(
            min_samples_leaf=5,
            random_state=42
        ).fit(X, y)
        
        preds = reg.predict(X)
        assert preds.shape == (50,)

    def test_score(self):
        """Test R^2 score."""
        np.random.seed(42)
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 2 * X.ravel() + 1
        
        reg = DecisionTreeRegressor(random_state=42).fit(X, y)
        score = reg.score(X, y)
        
        assert score > 0.9

    def test_multivariate(self):
        """Test multivariate regression."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + 2 * X[:, 1] - X[:, 2]
        
        reg = DecisionTreeRegressor(random_state=42).fit(X, y)
        preds = reg.predict(X)
        
        assert preds.shape == (100,)

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        reg = DecisionTreeRegressor()
        with pytest.raises(RuntimeError):
            reg.predict([[1, 2]])

    def test_shape_mismatch(self):
        """Test shape mismatch error."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([1, 2, 3])
        
        reg = DecisionTreeRegressor()
        with pytest.raises(ValueError):
            reg.fit(X, y)