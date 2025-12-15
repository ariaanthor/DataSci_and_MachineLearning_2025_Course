"""Unit tests for K-means clustering module."""

import numpy as np
import pytest

from rice_ml.unsupervised_learning.k_means_clustering import KMeans


class TestKMeans:
    """Tests for KMeans class."""

    def test_basic_fit(self):
        """Test basic fit."""
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        
        assert kmeans.labels_ is not None
        assert kmeans.cluster_centers_ is not None
        assert len(kmeans.labels_) == 6
        assert kmeans.cluster_centers_.shape == (2, 2)

    def test_two_clusters(self):
        """Test clustering into two well-separated groups."""
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        
        # First 3 points should be in one cluster, last 3 in another
        assert len(set(kmeans.labels_[:3])) == 1
        assert len(set(kmeans.labels_[3:])) == 1
        assert kmeans.labels_[0] != kmeans.labels_[3]

    def test_predict(self):
        """Test predict on new data."""
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        
        # New points near the clusters
        X_new = np.array([[0, 0], [12, 3]], dtype=float)
        labels = kmeans.predict(X_new)
        
        assert len(labels) == 2
        assert labels[0] != labels[1]

    def test_fit_predict(self):
        """Test fit_predict returns same labels as fit then predict."""
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
        
        kmeans = KMeans(n_clusters=2, random_state=0)
        labels = kmeans.fit_predict(X)
        
        assert np.array_equal(labels, kmeans.labels_)

    def test_inertia(self):
        """Test that inertia is computed."""
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        
        assert kmeans.inertia_ >= 0
        assert kmeans.inertia_ < float('inf')

    def test_n_iter(self):
        """Test that n_iter is tracked."""
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        
        assert kmeans.n_iter_ > 0

    def test_transform(self):
        """Test transform to cluster-distance space."""
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        distances = kmeans.transform(X)
        
        assert distances.shape == (6, 2)
        assert np.all(distances >= 0)

    def test_score(self):
        """Test score (negative inertia)."""
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        score = kmeans.score(X)
        
        assert score == -kmeans.inertia_

    def test_kmeans_plusplus_init(self):
        """Test k-means++ initialization."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [5, 5],
        ])
        
        kmeans = KMeans(n_clusters=2, init="k-means++", random_state=0).fit(X)
        
        assert kmeans.labels_ is not None
        assert len(set(kmeans.labels_)) == 2

    def test_random_init(self):
        """Test random initialization."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [5, 5],
        ])
        
        kmeans = KMeans(n_clusters=2, init="random", random_state=0).fit(X)
        
        assert kmeans.labels_ is not None

    def test_n_init(self):
        """Test multiple initializations."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        
        # More initializations should give same or better result
        kmeans_1 = KMeans(n_clusters=3, n_init=1, random_state=0).fit(X)
        kmeans_10 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
        
        assert kmeans_10.inertia_ <= kmeans_1.inertia_ * 1.1  # Allow some tolerance

    def test_convergence(self):
        """Test convergence with tolerance."""
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
        
        kmeans = KMeans(n_clusters=2, tol=1e-4, random_state=0).fit(X)
        
        # Should converge before max_iter for this simple case
        assert kmeans.n_iter_ < 300

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        kmeans = KMeans(n_clusters=2)
        with pytest.raises(RuntimeError):
            kmeans.predict([[1, 2]])

    def test_invalid_n_clusters(self):
        """Test invalid n_clusters."""
        with pytest.raises(ValueError):
            KMeans(n_clusters=0)

    def test_invalid_init(self):
        """Test invalid init method."""
        with pytest.raises(ValueError):
            KMeans(init="invalid")

    def test_insufficient_samples(self):
        """Test error when n_samples < n_clusters."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        
        kmeans = KMeans(n_clusters=5)
        with pytest.raises(ValueError):
            kmeans.fit(X)

    def test_reproducibility(self):
        """Test that random_state makes results reproducible."""
        X = np.random.randn(50, 2)
        
        kmeans1 = KMeans(n_clusters=3, random_state=42).fit(X)
        kmeans2 = KMeans(n_clusters=3, random_state=42).fit(X)
        
        assert np.array_equal(kmeans1.labels_, kmeans2.labels_)
        assert np.allclose(kmeans1.cluster_centers_, kmeans2.cluster_centers_)