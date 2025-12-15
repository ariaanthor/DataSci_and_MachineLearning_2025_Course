"""Unit tests for DBSCAN module."""

import numpy as np
import pytest

from rice_ml.unsupervised_learning.dbscan import DBSCAN


class TestDBSCAN:
    """Tests for DBSCAN class."""

    def test_basic_fit(self):
        """Test basic fit."""
        X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]], dtype=float)
        
        db = DBSCAN(eps=3, min_samples=2).fit(X)
        
        assert db.labels_ is not None
        assert len(db.labels_) == 6

    def test_two_clusters_and_noise(self):
        """Test detection of two clusters and noise."""
        X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]], dtype=float)
        
        db = DBSCAN(eps=3, min_samples=2).fit(X)
        
        # Should find 2 clusters and 1 noise point
        unique_labels = set(db.labels_)
        assert -1 in unique_labels  # Noise
        assert len(unique_labels) == 3  # 2 clusters + noise

    def test_fit_predict(self):
        """Test fit_predict method."""
        X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]], dtype=float)
        
        db = DBSCAN(eps=3, min_samples=2)
        labels = db.fit_predict(X)
        
        assert np.array_equal(labels, db.labels_)

    def test_core_samples(self):
        """Test core sample detection."""
        X = np.array([[0, 0], [0.5, 0], [1, 0], [5, 0]], dtype=float)
        
        db = DBSCAN(eps=1.0, min_samples=2).fit(X)
        
        assert db.core_sample_indices_ is not None
        assert len(db.core_sample_indices_) > 0

    def test_components(self):
        """Test components attribute."""
        X = np.array([[0, 0], [0.5, 0], [1, 0], [5, 0]], dtype=float)
        
        db = DBSCAN(eps=1.0, min_samples=2).fit(X)
        
        assert db.components_ is not None
        # Components should be the core samples
        assert len(db.components_) == len(db.core_sample_indices_)

    def test_all_noise(self):
        """Test when all points are noise."""
        X = np.array([[0, 0], [10, 10], [20, 20]], dtype=float)
        
        db = DBSCAN(eps=1.0, min_samples=2).fit(X)
        
        # All points should be noise
        assert np.all(db.labels_ == -1)

    def test_all_one_cluster(self):
        """Test when all points are in one cluster."""
        X = np.array([[0, 0], [0.1, 0], [0.2, 0], [0.3, 0]], dtype=float)
        
        db = DBSCAN(eps=0.5, min_samples=2).fit(X)
        
        # All points should be in the same cluster
        assert len(set(db.labels_)) == 1
        assert db.labels_[0] >= 0  # Not noise

    def test_eps_effect(self):
        """Test effect of eps parameter."""
        X = np.array([[0, 0], [1, 0], [2, 0], [10, 0], [11, 0], [12, 0]], dtype=float)
        
        # Small eps: more clusters
        db_small = DBSCAN(eps=1.5, min_samples=2).fit(X)
        
        # Large eps: fewer clusters
        db_large = DBSCAN(eps=15, min_samples=2).fit(X)
        
        n_clusters_small = len(set(db_small.labels_) - {-1})
        n_clusters_large = len(set(db_large.labels_) - {-1})
        
        assert n_clusters_small >= n_clusters_large

    def test_min_samples_effect(self):
        """Test effect of min_samples parameter."""
        X = np.array([[0, 0], [0.1, 0], [0.2, 0], [10, 10]], dtype=float)
        
        # Low min_samples: point at (10,10) might form cluster
        db_low = DBSCAN(eps=1.0, min_samples=1).fit(X)
        
        # High min_samples: need more neighbors
        db_high = DBSCAN(eps=1.0, min_samples=3).fit(X)
        
        # With higher min_samples, we should have more noise
        n_noise_low = np.sum(db_low.labels_ == -1)
        n_noise_high = np.sum(db_high.labels_ == -1)
        assert n_noise_high >= n_noise_low

    def test_manhattan_metric(self):
        """Test with Manhattan distance."""
        X = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], dtype=float)
        
        db = DBSCAN(eps=2.0, min_samples=2, metric="manhattan").fit(X)
        
        assert db.labels_ is not None

    def test_euclidean_metric(self):
        """Test with Euclidean distance."""
        X = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], dtype=float)
        
        db = DBSCAN(eps=1.5, min_samples=2, metric="euclidean").fit(X)
        
        assert db.labels_ is not None

    def test_invalid_eps(self):
        """Test invalid eps."""
        with pytest.raises(ValueError):
            DBSCAN(eps=0)
        
        with pytest.raises(ValueError):
            DBSCAN(eps=-1)

    def test_invalid_min_samples(self):
        """Test invalid min_samples."""
        with pytest.raises(ValueError):
            DBSCAN(min_samples=0)

    def test_invalid_metric(self):
        """Test invalid metric."""
        with pytest.raises(ValueError):
            DBSCAN(metric="invalid")

    def test_non_square_matrix(self):
        """Test with non-square input (feature matrix)."""
        X = np.random.randn(50, 3)
        
        db = DBSCAN(eps=1.0, min_samples=5).fit(X)
        
        assert len(db.labels_) == 50

    def test_single_point(self):
        """Test with single point."""
        X = np.array([[0, 0]], dtype=float)
        
        db = DBSCAN(eps=1.0, min_samples=1).fit(X)
        
        # Single point with min_samples=1 should be in its own cluster
        assert db.labels_[0] >= 0

    def test_reproducibility(self):
        """Test that results are deterministic."""
        X = np.random.randn(50, 2)
        
        db1 = DBSCAN(eps=1.0, min_samples=3).fit(X)
        db2 = DBSCAN(eps=1.0, min_samples=3).fit(X)
        
        assert np.array_equal(db1.labels_, db2.labels_)

    def test_dense_cluster(self):
        """Test with a dense cluster."""
        np.random.seed(42)
        # One dense cluster
        X = np.random.randn(50, 2) * 0.1
        # Add some noise
        X = np.vstack([X, np.random.randn(5, 2) * 10])
        
        db = DBSCAN(eps=0.5, min_samples=5).fit(X)
        
        # Most points should be in one cluster
        n_in_cluster = np.sum(db.labels_ >= 0)
        assert n_in_cluster >= 40