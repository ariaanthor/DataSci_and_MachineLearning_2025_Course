import numpy as np
import pytest

from rice_ml.processing.post_processing import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    mse,
    rmse,
    mae,
    r2_score,
)


# -------------------- Classification: binary --------------------

class TestBinaryClassificationMetrics:
    """Tests for binary classification metrics."""

    def test_basic_metrics(self):
        """Test basic binary classification metrics."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        
        assert accuracy_score(y_true, y_pred) == 0.75
        # Only one positive predicted and it's correct
        assert precision_score(y_true, y_pred, average="binary") == 1.0
        assert recall_score(y_true, y_pred, average="binary") == 0.5
        assert f1_score(y_true, y_pred, average="binary") == 2 * 1.0 * 0.5 / (1.0 + 0.5)
        
        cm = confusion_matrix(y_true, y_pred)
        assert cm.tolist() == [[2, 0], [1, 1]]

    def test_roc_auc(self):
        """Test ROC AUC score."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        assert round(roc_auc_score(y_true, scores), 2) == 0.75

    def test_log_loss_binary(self):
        """Test log loss for binary classification."""
        probs = np.array([0.1, 0.9])  # prob of class 1
        y = np.array([0, 1])
        ll = log_loss(y, probs)
        assert np.isclose(ll, -np.log(0.9))


class TestMulticlassClassificationMetrics:
    """Tests for multiclass classification metrics."""

    def test_log_loss_multiclass(self):
        """Test log loss for multiclass one-hot."""
        y_true = np.array([0, 1, 2])
        probs = np.eye(3)
        assert log_loss(y_true, probs) == 0.0

    def test_macro_micro(self):
        """Test macro and micro averaging."""
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([0, 2, 2, 1])
        
        assert accuracy_score(y_true, y_pred) == 0.5
        assert precision_score(y_true, y_pred, average="macro") == 0.5
        assert recall_score(y_true, y_pred, average="macro") == 0.5
        assert f1_score(y_true, y_pred, average="macro") == 0.5
        
        # micro = accuracy for single-label multiclass
        assert precision_score(y_true, y_pred, average="micro") == 0.5
        assert recall_score(y_true, y_pred, average="micro") == 0.5
        assert f1_score(y_true, y_pred, average="micro") == 0.5
        
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_confusion_with_custom_labels(self):
        """Test confusion matrix with custom labels ignoring unknowns."""
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([0, 3, 2, 1])  # 3 not in labels below -> ignored
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        # The prediction "3" for a true "1" is ignored -> no column accumulates it
        assert cm.tolist() == [[1, 0, 0],
                               [0, 0, 0],
                               [0, 1, 1]]


class TestBinaryMetricErrors:
    """Tests for binary metric error cases."""

    def test_precision_not_binary(self):
        """Test precision error with non-binary classes."""
        with pytest.raises(ValueError):
            precision_score([0, 1, 2], [0, 1, 2], average="binary")

    def test_roc_auc_single_class(self):
        """Test ROC AUC error with single class."""
        with pytest.raises(ValueError):
            roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3])

    def test_log_loss_invalid_probs(self):
        """Test log loss error with invalid probabilities."""
        with pytest.raises(ValueError):
            log_loss([0, 1], np.array([1.2, 0.5]))


# --------------------------- Regression ---------------------------

class TestRegressionMetrics:
    """Tests for regression metrics."""

    def test_basic_metrics(self):
        """Test basic regression metrics."""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        
        assert mse(y_true, y_pred) == 0.375
        assert round(rmse(y_true, y_pred), 6) == 0.612372
        assert mae(y_true, y_pred) == 0.5
        assert round(r2_score(y_true, y_pred), 6) == 0.948608

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        
        assert mse(y_true, y_pred) == 0.0
        assert rmse(y_true, y_pred) == 0.0
        assert mae(y_true, y_pred) == 0.0
        assert r2_score(y_true, y_pred) == 1.0


class TestRegressionMetricErrors:
    """Tests for regression metric error cases."""

    def test_shape_mismatch(self):
        """Test shape mismatch error."""
        with pytest.raises(ValueError):
            mse([1, 2], [1])

    def test_type_error(self):
        """Test type error with non-numeric input."""
        with pytest.raises(TypeError):
            mae(["a", "b"], [1, 2])

    def test_r2_constant_y(self):
        """Test R^2 with constant y."""
        # If perfect predictions with constant y, R^2 = 1.0
        y_const = np.array([5.0, 5.0, 5.0])
        y_pred_perfect = np.array([5.0, 5.0, 5.0])
        assert r2_score(y_const, y_pred_perfect) == 1.0
        
        # If imperfect predictions with constant y, raise error
        y_pred_bad = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError):
            r2_score(y_const, y_pred_bad)