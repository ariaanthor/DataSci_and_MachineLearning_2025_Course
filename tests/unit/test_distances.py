import pytest
import numpy as np

from rice_ml.supervised_learning.distance_metrics import euclidean_distance, manhattan_distance


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_basic(self):
        """Test basic Euclidean distance."""
        assert euclidean_distance(np.array([0, 0]), np.array([3, 4])) == 5.0
        assert euclidean_distance([1, 2, 3], [1, 2, 3]) == 0.0

    def test_different_values(self):
        """Test with different values."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 6.0, 3.0])
        expected = np.sqrt(9 + 16 + 0)  # 5.0
        assert np.isclose(euclidean_distance(a, b), expected)

    def test_negative_values(self):
        """Test with negative values."""
        a = np.array([-1, -2])
        b = np.array([1, 2])
        expected = np.sqrt(4 + 16)
        assert np.isclose(euclidean_distance(a, b), expected)


class TestManhattanDistance:
    """Tests for manhattan_distance function."""

    def test_basic(self):
        """Test basic Manhattan distance."""
        assert manhattan_distance(np.array([1, 2, 3]), np.array([4, 0, 3])) == 5.0
        assert manhattan_distance([0, 0], [0, 0]) == 0.0

    def test_different_values(self):
        """Test with different values."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 6.0, 3.0])
        expected = 3 + 4 + 0  # 7.0
        assert manhattan_distance(a, b) == expected

    def test_negative_values(self):
        """Test with negative values."""
        a = np.array([-1, -2])
        b = np.array([1, 2])
        expected = 2 + 4  # 6.0
        assert manhattan_distance(a, b) == expected


class TestDistanceValidation:
    """Tests for input validation."""

    def test_invalid_shape(self):
        """Test error on invalid shape."""
        with pytest.raises(ValueError):
            euclidean_distance(np.array([[1, 2], [3, 4]]), np.array([1, 2]))
        with pytest.raises(ValueError):
            manhattan_distance(np.array([1, 2, 3]), np.array([1, 2]))

    def test_type_validation(self):
        """Test error on non-numeric types."""
        with pytest.raises(TypeError):
            euclidean_distance(["a", "b"], [1, 2])
        with pytest.raises(TypeError):
            manhattan_distance([1, 2], ["x", "y"])

    def test_symmetry_and_nonnegative(self):
        """Test symmetry and non-negativity properties."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        
        # Symmetry
        assert euclidean_distance(a, b) == euclidean_distance(b, a)
        assert manhattan_distance(a, b) == manhattan_distance(b, a)
        
        # Non-negativity
        assert euclidean_distance(a, b) >= 0
        assert manhattan_distance(a, b) >= 0