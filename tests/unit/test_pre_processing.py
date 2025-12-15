import numpy as np
import pytest
from collections import Counter

from rice_ml.processing.pre_processing import (
    standardize,
    minmax_scale,
    maxabs_scale,
    l2_normalize_rows,
    train_test_split,
    train_val_test_split,
)


# ---------------------- Scaling & Normalization ----------------------

class TestStandardize:
    """Tests for standardize function."""

    def test_basic_and_params(self):
        """Test basic standardization and parameter return."""
        X = np.array([[1., 2.], [3., 2.], [5., 2.]])
        Z, params = standardize(X, return_params=True)
        
        assert Z.shape == X.shape
        # Column 1 has variance, column 2 is constant
        assert np.allclose(Z[:, 1], 0.0)
        assert params["scale"][1] == 1.0  # zero variance handled
        # Centered means ~0
        assert np.allclose(Z.mean(axis=0), 0.0)

    def test_no_std_or_mean(self):
        """Test standardization without std or mean."""
        X = np.array([[1., 2.], [3., 4.]])
        
        Z = standardize(X, with_mean=False, with_std=False)
        assert np.allclose(Z, X)
        
        Z = standardize(X, with_mean=True, with_std=False)
        assert not np.allclose(Z, X)
        assert np.allclose(Z.mean(axis=0), 0.0)


class TestMinmaxScale:
    """Tests for minmax_scale function."""

    def test_range_and_params(self):
        """Test min-max scaling with custom range."""
        X = np.array([[0., 10.], [5., 10.], [10., 10.]])
        X2, params = minmax_scale(X, feature_range=(2, 3), return_params=True)
        
        assert X2.shape == X.shape
        # First feature maps 0->2, 10->3
        assert np.allclose(X2[:, 0], [2.0, 2.5, 3.0])
        # Second feature zero-range -> mapped to lower bound
        assert np.allclose(X2[:, 1], 2.0)
        assert params["feature_range"] == (2.0, 3.0)
        assert params["scale"][1] == 1.0


class TestMaxabsScale:
    """Tests for maxabs_scale function."""

    def test_basic(self):
        """Test basic max-abs scaling."""
        X = np.array([[-2., 0.], [1., 0.], [2., 0.]])
        X2, params = maxabs_scale(X, return_params=True)
        
        assert np.allclose(X2[:, 0], [-1.0, 0.5, 1.0])
        assert np.allclose(X2[:, 1], [0.0, 0.0, 0.0])
        assert params["scale"][1] == 1.0


class TestL2NormalizeRows:
    """Tests for l2_normalize_rows function."""

    def test_behavior(self):
        """Test L2 normalization behavior."""
        X = np.array([[3., 4.], [0., 0.]])
        Xn = l2_normalize_rows(X)
        
        assert np.isclose(np.linalg.norm(Xn[0]), 1.0)
        assert np.allclose(Xn[1], [0.0, 0.0])

    def test_invalid_eps(self):
        """Test error with invalid eps."""
        X = np.array([[3., 4.], [0., 0.]])
        with pytest.raises(ValueError):
            l2_normalize_rows(X, eps=0.0)


class TestScalerInputValidation:
    """Tests for scaler input validation."""

    def test_not_2d(self):
        """Test error with non-2D input."""
        with pytest.raises(ValueError):
            standardize(np.array([1., 2., 3.]))

    def test_non_numeric(self):
        """Test error with non-numeric input."""
        with pytest.raises(TypeError):
            standardize([["a", "b"], ["c", "d"]])

    def test_empty(self):
        """Test error with empty input."""
        with pytest.raises(ValueError):
            minmax_scale(np.empty((0, 2)))

    def test_invalid_feature_range(self):
        """Test error with invalid feature range."""
        with pytest.raises(ValueError):
            minmax_scale(np.ones((2, 2)), feature_range=(1, 1))


# ---------------------- Splitting ----------------------

class TestTrainTestSplit:
    """Tests for train_test_split function."""

    def test_shapes_and_determinism(self):
        """Test output shapes and determinism."""
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        
        X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, test_size=0.3, random_state=42)
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.3, random_state=42)
        
        assert X_tr1.shape == (35, 2)
        assert X_te1.shape == (15, 2)
        assert np.array_equal(y_tr1, y_tr2)
        assert np.array_equal(X_te1, X_te2)

    def test_no_shuffle(self):
        """Test split without shuffling."""
        X = np.arange(100).reshape(50, 2)
        X_tr, X_te = train_test_split(X, test_size=0.2, shuffle=False)
        
        assert X_tr.shape == (40, 2)
        assert X_te.shape == (10, 2)

    def test_stratify(self):
        """Test stratified split."""
        X = np.arange(60).reshape(30, 2)
        y = np.array([0, 1, 2] * 10)
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        
        # Each class appears in both splits
        assert set(np.unique(y_tr)) == set(np.unique(y_te))
        
        # Proportions roughly preserved
        c_full = Counter(y)
        c_te = Counter(y_te)
        for k in c_full:
            expected_te = round(0.3 * c_full[k])
            assert abs(c_te[k] - expected_te) <= 1


class TestTrainValTestSplit:
    """Tests for train_val_test_split function."""

    def test_shapes_and_stratify(self):
        """Test output shapes with stratification."""
        X = np.arange(90).reshape(45, 2)
        y = np.array([0, 1, 2] * 15)
        
        parts = train_val_test_split(X, y, val_size=0.2, test_size=0.2, stratify=y, random_state=123)
        X_tr, X_va, X_te, y_tr, y_va, y_te = parts
        
        assert X_tr.shape == (27, 2)
        assert X_va.shape == (9, 2)
        assert X_te.shape == (9, 2)
        assert set(np.unique(y_tr)) == set(np.unique(y_va)) == set(np.unique(y_te))

    def test_without_y(self):
        """Test split without y."""
        X = np.arange(30).reshape(15, 2)
        X_tr, X_va, X_te = train_val_test_split(X, val_size=0.2, test_size=0.2, random_state=7)
        
        assert X_tr.shape == (9, 2)
        assert X_va.shape == (3, 2)
        assert X_te.shape == (3, 2)


class TestSplitInputValidation:
    """Tests for split input validation."""

    def test_not_2d(self):
        """Test error with non-2D X."""
        X = np.arange(10)
        y = np.arange(10)
        with pytest.raises(ValueError):
            train_test_split(X, y)

    def test_shape_mismatch(self):
        """Test error with shape mismatch."""
        X = np.arange(20).reshape(10, 2)
        y = np.arange(9)
        with pytest.raises(ValueError):
            train_test_split(X, y)

    def test_invalid_test_size(self):
        """Test error with invalid test size."""
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=1.5)

    def test_invalid_random_state(self):
        """Test error with invalid random state."""
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        with pytest.raises(TypeError):
            train_test_split(X, y, random_state="seed")

    def test_invalid_val_test_sum(self):
        """Test error when val_size + test_size >= 1."""
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        with pytest.raises(ValueError):
            train_val_test_split(X, y, val_size=0.6, test_size=0.5)

    def test_negative_val_size(self):
        """Test error with negative val_size."""
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        with pytest.raises(ValueError):
            train_val_test_split(X, y, val_size=-0.1, test_size=0.2)