import numpy as np
import pytest

from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor


# ------------------------ Classifier ------------------------

class TestKNNClassifier:
    """Tests for KNNClassifier class."""

    def test_basic_predict_and_proba_uniform_euclidean(self):
        """Test basic prediction with uniform weights and Euclidean distance."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        
        clf = KNNClassifier(n_neighbors=3, metric="euclidean", weights="uniform").fit(X, y)
        preds = clf.predict([[0.1, 0.1], [0.9, 0.9]])
        
        assert preds.tolist() == [0, 1]
        
        proba = clf.predict_proba([[0.1, 0.1], [0.9, 0.9]])
        # Rows sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
        # Class order is sorted(unique) = [0, 1]
        assert (proba.argmax(axis=1) == preds).all()

    def test_manhattan_distance_weighted(self):
        """Test Manhattan distance with distance weighting."""
        X = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=float)
        y = np.array(["A", "A", "B", "B"], dtype=object)
        
        # Query near (0, 0) should favor A
        clf = KNNClassifier(n_neighbors=3, metric="manhattan", weights="distance").fit(X, y)
        pred = clf.predict([[0.1, 0.2]])
        
        assert pred.tolist() == ["A"]
        
        # predict_proba should be concentrated on A
        p = clf.predict_proba([[0.1, 0.2]])[0]
        assert p[0] > p[1]  # classes_ sorted -> ["A", "B"]

    def test_errors_and_kneighbors(self):
        """Test error handling and kneighbors method."""
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        y = np.array([0, 1, 1])
        
        clf = KNNClassifier(n_neighbors=2).fit(X, y)
        
        # Wrong feature count
        with pytest.raises(ValueError):
            clf.predict([[0.0, 0.0, 0.0]])
        
        # kneighbors returns shapes (nq, k)
        d, idx = clf.kneighbors([[1.0, 1.0]])
        assert d.shape == (1, 2) and idx.shape == (1, 2)

    def test_score_accuracy(self):
        """Test accuracy score."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        
        clf = KNNClassifier(n_neighbors=1).fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_zero_distance_with_distance_weights(self):
        """Test zero distance handling with distance weights."""
        # Exact duplicate point in training -> zero distance handling
        X = np.array([[0, 0], [1, 1], [0, 0]], dtype=float)
        y = np.array([0, 1, 0])
        
        clf = KNNClassifier(n_neighbors=2, weights="distance").fit(X, y)
        
        # Query exactly matches (0, 0); only zero-distance neighbors count -> both class 0
        pred = clf.predict([[0, 0]])
        assert pred.tolist() == [0]
        
        p = clf.predict_proba([[0, 0]])[0]
        # Full mass on class 0
        assert np.isclose(p[0], 1.0)

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        clf = KNNClassifier()
        with pytest.raises(RuntimeError):
            clf.predict([[1, 2]])


# ------------------------ Regressor ------------------------

class TestKNNRegressor:
    """Tests for KNNRegressor class."""

    def test_basic_predict_and_score(self):
        """Test basic prediction and score."""
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0.0, 1.0, 1.5, 3.0])
        
        reg = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
        pred = reg.predict([[1.5]])[0]
        assert 1.2 < pred < 1.3
        
        # Perfect fit at training points with k=1
        reg2 = KNNRegressor(n_neighbors=1).fit(X, y)
        assert reg2.score(X, y) == 1.0

    def test_input_errors(self):
        """Test input validation errors."""
        X = np.array([[0], [1], [2]], dtype=float)
        y = np.array([0.0, 1.0, 2.0])
        
        # n_neighbors > n_samples
        with pytest.raises(ValueError):
            KNNRegressor(n_neighbors=5).fit(X, y)
        
        # Non-numeric y
        with pytest.raises(TypeError):
            KNNRegressor(n_neighbors=1).fit(X, np.array(["a", "b", "c"], dtype=object))

    def test_constant_y_score(self):
        """Test score with constant y."""
        X = np.array([[0], [1], [2]], dtype=float)
        y = np.array([5.0, 5.0, 5.0])
        
        reg = KNNRegressor(n_neighbors=1).fit(X, y)
        
        # Using training X with k=1 => perfect fit
        assert reg.score(X, y) == 1.0
        
        # Perturb X slightly to avoid exact matches -> should raise
        with pytest.raises(ValueError):
            reg.score(X + 0.1, y)

    def test_uniform_weights(self):
        """Test uniform weights."""
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0.0, 1.0, 2.0, 3.0])
        
        reg = KNNRegressor(n_neighbors=2, weights="uniform").fit(X, y)
        pred = reg.predict([[1.5]])[0]
        
        # With uniform weights, prediction is mean of 2 nearest
        assert np.isclose(pred, 1.5, atol=0.01)

    def test_manhattan_metric(self):
        """Test Manhattan distance metric."""
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        y = np.array([0.0, 1.0, 2.0])
        
        reg = KNNRegressor(n_neighbors=2, metric="manhattan").fit(X, y)
        pred = reg.predict([[1, 1]])
        
        assert pred.shape == (1,)