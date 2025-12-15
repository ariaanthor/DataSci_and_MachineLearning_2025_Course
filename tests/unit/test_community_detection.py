"""Unit tests for community detection module."""

import numpy as np
import pytest

from rice_ml.unsupervised_learning.community_detection import SpectralClustering, LabelPropagation


class TestSpectralClustering:
    """Tests for SpectralClustering class."""

    def test_basic_fit(self):
        """Test basic fit on adjacency matrix."""
        # Two connected components
        A = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0]
        ], dtype=float)
        
        sc = SpectralClustering(n_clusters=2, random_state=0).fit(A)
        
        assert sc.labels_ is not None
        assert len(sc.labels_) == 5

    def test_two_components(self):
        """Test detection of two connected components."""
        A = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0]
        ], dtype=float)
        
        sc = SpectralClustering(n_clusters=2, random_state=0).fit(A)
        
        # First 3 nodes should be in one cluster, last 2 in another
        assert sc.labels_[0] == sc.labels_[1] == sc.labels_[2]
        assert sc.labels_[3] == sc.labels_[4]
        assert sc.labels_[0] != sc.labels_[3]

    def test_fit_predict(self):
        """Test fit_predict method."""
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]
        ], dtype=float)
        
        sc = SpectralClustering(n_clusters=2, random_state=0)
        labels = sc.fit_predict(A)
        
        assert np.array_equal(labels, sc.labels_)

    def test_affinity_matrix_stored(self):
        """Test that affinity matrix is stored."""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        
        sc = SpectralClustering(n_clusters=2, random_state=0).fit(A)
        
        assert sc.affinity_matrix_ is not None
        assert sc.affinity_matrix_.shape == (2, 2)

    def test_rbf_affinity(self):
        """Test RBF affinity on feature matrix."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(10, 2) + [0, 0],
            np.random.randn(10, 2) + [5, 5],
        ])
        
        sc = SpectralClustering(
            n_clusters=2,
            affinity="rbf",
            random_state=0
        ).fit(X)
        
        assert len(sc.labels_) == 20
        assert len(set(sc.labels_)) == 2

    def test_n_init(self):
        """Test n_init parameter."""
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=float)
        
        sc = SpectralClustering(n_clusters=2, n_init=5, random_state=0).fit(A)
        
        assert sc.labels_ is not None

    def test_invalid_n_clusters(self):
        """Test invalid n_clusters."""
        with pytest.raises(ValueError):
            SpectralClustering(n_clusters=0)

    def test_invalid_affinity(self):
        """Test invalid affinity."""
        with pytest.raises(ValueError):
            SpectralClustering(affinity="invalid")

    def test_non_square_precomputed(self):
        """Test error for non-square precomputed matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        
        sc = SpectralClustering(n_clusters=2, affinity="precomputed")
        with pytest.raises(ValueError):
            sc.fit(A)

    def test_reproducibility(self):
        """Test reproducibility with random_state."""
        A = np.array([
            [0, 1, 0.5, 0],
            [1, 0, 0.5, 0],
            [0.5, 0.5, 0, 1],
            [0, 0, 1, 0]
        ], dtype=float)
        
        sc1 = SpectralClustering(n_clusters=2, random_state=42).fit(A)
        sc2 = SpectralClustering(n_clusters=2, random_state=42).fit(A)
        
        assert np.array_equal(sc1.labels_, sc2.labels_)


class TestLabelPropagation:
    """Tests for LabelPropagation class."""

    def test_basic_fit(self):
        """Test basic fit on adjacency matrix."""
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=float)
        
        lp = LabelPropagation(random_state=0).fit(A)
        
        assert lp.labels_ is not None
        assert len(lp.labels_) == 4

    def test_connected_graph(self):
        """Test on fully connected graph."""
        A = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]
        ], dtype=float)
        
        lp = LabelPropagation(random_state=0).fit(A)
        
        # All nodes should end up in same community
        assert len(set(lp.labels_)) == 1

    def test_two_components(self):
        """Test detection of two components."""
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=float)
        
        lp = LabelPropagation(random_state=0).fit(A)
        
        # Should detect 2 communities
        assert lp.labels_[0] == lp.labels_[1]
        assert lp.labels_[2] == lp.labels_[3]
        assert lp.labels_[0] != lp.labels_[2]

    def test_fit_predict(self):
        """Test fit_predict method."""
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]
        ], dtype=float)
        
        lp = LabelPropagation(random_state=0)
        labels = lp.fit_predict(A)
        
        assert np.array_equal(labels, lp.labels_)

    def test_n_iter_tracked(self):
        """Test that n_iter is tracked."""
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=float)
        
        lp = LabelPropagation(random_state=0).fit(A)
        
        assert lp.n_iter_ > 0

    def test_max_iter(self):
        """Test max_iter parameter."""
        A = np.random.rand(10, 10)
        A = (A + A.T) / 2  # Symmetrize
        np.fill_diagonal(A, 0)
        
        lp = LabelPropagation(max_iter=5, random_state=0).fit(A)
        
        assert lp.n_iter_ <= 5

    def test_weighted_edges(self):
        """Test with weighted adjacency matrix."""
        A = np.array([
            [0, 10, 0.1, 0],
            [10, 0, 0.1, 0],
            [0.1, 0.1, 0, 10],
            [0, 0, 10, 0]
        ], dtype=float)
        
        lp = LabelPropagation(random_state=0).fit(A)
        
        # Strong edges should keep nodes together
        assert lp.labels_[0] == lp.labels_[1]
        assert lp.labels_[2] == lp.labels_[3]

    def test_labels_are_consecutive(self):
        """Test that labels are consecutive integers."""
        A = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=float)
        
        lp = LabelPropagation(random_state=0).fit(A)
        
        # Labels should be 0, 1, 2, ... (consecutive)
        unique_labels = np.unique(lp.labels_)
        assert np.array_equal(unique_labels, np.arange(len(unique_labels)))

    def test_invalid_max_iter(self):
        """Test invalid max_iter."""
        with pytest.raises(ValueError):
            LabelPropagation(max_iter=0)

    def test_non_square_matrix(self):
        """Test error for non-square matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        
        lp = LabelPropagation()
        with pytest.raises(ValueError):
            lp.fit(A)

    def test_isolated_nodes(self):
        """Test with isolated nodes."""
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],  # Isolated node
            [0, 0, 0, 0]   # Isolated node
        ], dtype=float)
        
        lp = LabelPropagation(random_state=0).fit(A)
        
        # Should still produce labels for all nodes
        assert len(lp.labels_) == 4