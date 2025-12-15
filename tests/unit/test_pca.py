"""Unit tests for PCA module."""

import numpy as np
import pytest

from rice_ml.unsupervised_learning.pca import PCA


class TestPCA:
    """Tests for PCA class."""

    def test_basic_fit_transform(self):
        """Test basic fit and transform."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        pca = PCA(n_components=2).fit(X)
        X_transformed = pca.transform(X)
        
        assert X_transformed.shape == (3, 2)

    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)
        
        assert X_transformed.shape == (3, 2)

    def test_explained_variance_ratio(self):
        """Test that explained variance ratio is computed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        pca = PCA(n_components=3).fit(X)
        
        assert pca.explained_variance_ratio_ is not None
        assert len(pca.explained_variance_ratio_) == 3
        # Ratios should be non-negative and sum to <= 1
        assert np.all(pca.explained_variance_ratio_ >= 0)
        assert np.sum(pca.explained_variance_ratio_) <= 1.0 + 1e-10

    def test_components(self):
        """Test that components are orthogonal."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        pca = PCA(n_components=3).fit(X)
        
        assert pca.components_.shape == (3, 5)
        # Components should be approximately orthogonal
        inner_products = pca.components_ @ pca.components_.T
        off_diagonal = inner_products - np.eye(3)
        assert np.allclose(off_diagonal, 0, atol=1e-10)

    def test_mean(self):
        """Test that mean is computed."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        
        pca = PCA(n_components=2).fit(X)
        
        assert pca.mean_ is not None
        assert np.allclose(pca.mean_, [3, 4])

    def test_inverse_transform(self):
        """Test inverse transform."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=5).fit(X)
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Full reconstruction should be exact
        assert np.allclose(X, X_reconstructed)

    def test_partial_reconstruction(self):
        """Test partial reconstruction with fewer components."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=2).fit(X)
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Partial reconstruction should be close but not exact
        assert X_reconstructed.shape == X.shape
        # Should have some reconstruction error
        error = np.mean((X - X_reconstructed) ** 2)
        assert error > 0

    def test_n_components_none(self):
        """Test with n_components=None (keep all)."""
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=None).fit(X)
        
        assert pca.n_components_ == 5
        assert pca.components_.shape == (5, 5)

    def test_n_components_float(self):
        """Test with n_components as float (variance ratio)."""
        np.random.seed(42)
        # Create data where first component explains most variance
        X = np.random.randn(100, 5)
        X[:, 0] *= 10  # Make first feature have high variance
        
        pca = PCA(n_components=0.9).fit(X)
        
        # Should select enough components to explain 90% variance
        assert pca.n_components_ >= 1
        assert np.sum(pca.explained_variance_ratio_) >= 0.9

    def test_singular_values(self):
        """Test that singular values are computed."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=3).fit(X)
        
        assert pca.singular_values_ is not None
        assert len(pca.singular_values_) == 3
        # Singular values should be non-negative and sorted descending
        assert np.all(pca.singular_values_ >= 0)
        assert np.all(np.diff(pca.singular_values_) <= 0)

    def test_get_covariance(self):
        """Test covariance estimation."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        
        pca = PCA(n_components=3).fit(X)
        cov = pca.get_covariance()
        
        assert cov.shape == (3, 3)
        # Covariance should be symmetric
        assert np.allclose(cov, cov.T)

    def test_score(self):
        """Test score method."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=3).fit(X)
        score = pca.score(X)
        
        # Score is negative reconstruction error
        assert score <= 0

    def test_not_fitted_error(self):
        """Test error when transforming without fitting."""
        pca = PCA(n_components=2)
        with pytest.raises(RuntimeError):
            pca.transform([[1, 2, 3]])

    def test_feature_mismatch(self):
        """Test error when transform has wrong features."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        pca = PCA(n_components=2).fit(X)
        
        with pytest.raises(ValueError):
            pca.transform([[1, 2]])

    def test_invalid_n_components_int(self):
        """Test invalid n_components (negative)."""
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError):
            PCA(n_components=-1).fit(X)

    def test_invalid_n_components_float(self):
        """Test invalid n_components float."""
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError):
            PCA(n_components=1.5).fit(X)

    def test_dimensionality_reduction(self):
        """Test actual dimensionality reduction."""
        np.random.seed(42)
        # Create correlated data
        X = np.random.randn(100, 2)
        X = np.column_stack([X, X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]])
        
        pca = PCA(n_components=2).fit(X)
        X_reduced = pca.transform(X)
        
        # Should capture most variance with 2 components
        total_var = np.sum(pca.explained_variance_ratio_)
        assert total_var > 0.9