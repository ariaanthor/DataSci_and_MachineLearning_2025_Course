import numpy as np
import pytest


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def binary_classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(30, 2) + [0, 0],
        np.random.randn(30, 2) + [3, 3],
    ])
    y = np.array([0] * 30 + [1] * 30)
    return X, y


@pytest.fixture
def multiclass_classification_data():
    """Generate simple multiclass classification data."""
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(20, 2) + [0, 0],
        np.random.randn(20, 2) + [4, 0],
        np.random.randn(20, 2) + [2, 4],
    ])
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    return X, y


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X.ravel() + 1 + 0.1 * np.random.randn(50)
    return X, y


@pytest.fixture
def clustering_data():
    """Generate data with clear clusters."""
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(30, 2) * 0.5 + [0, 0],
        np.random.randn(30, 2) * 0.5 + [5, 5],
        np.random.randn(30, 2) * 0.5 + [5, 0],
    ])
    return X