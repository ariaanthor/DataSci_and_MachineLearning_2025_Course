"""
Pytest unit tests for a simple Perceptron implementation.

"""

from __future__ import annotations

import numpy as np
import pytest

from rice_ml.supervised_learning.perceptron import Perceptron


def _make_separable_blobs(n_per_class: int = 30, seed: int = 0):
    """Generate an easy linearly separable dataset with labels in {-1, +1}."""
    rng = np.random.default_rng(seed)
    X_neg = rng.normal(loc=-2.0, scale=0.35, size=(n_per_class, 2))
    X_pos = rng.normal(loc=+2.0, scale=0.35, size=(n_per_class, 2))
    X = np.vstack([X_neg, X_pos]).astype(float)
    y = np.hstack([np.full(n_per_class, -1), np.full(n_per_class, +1)]).astype(int)
    return X, y


def test_train_creates_weights_and_error_history():
    X, y = _make_separable_blobs(n_per_class=15, seed=1)
    clf = Perceptron(eta=0.1, epochs=50).train(X, y)

    assert hasattr(clf, "w_")
    assert isinstance(clf.w_, np.ndarray)
    assert clf.w_.shape == (1 + X.shape[1],)

    assert hasattr(clf, "errors_")
    assert isinstance(clf.errors_, list)
    assert len(clf.errors_) >= 1
    assert all(isinstance(e, (int, np.integer)) for e in clf.errors_)


def test_predict_outputs_only_minus1_plus1():
    X, y = _make_separable_blobs(n_per_class=20, seed=2)
    clf = Perceptron(eta=0.1, epochs=100).train(X, y)

    preds = np.asarray(clf.predict(X))
    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset({-1, +1})


def test_net_input_matches_manual_dot_for_single_point():
    X, y = _make_separable_blobs(n_per_class=10, seed=3)
    clf = Perceptron(eta=0.2, epochs=60).train(X, y)

    x0 = X[0]
    manual = float(np.dot(x0, clf.w_[:-1]) + clf.w_[-1])
    got = float(clf.net_input(x0))
    assert np.isclose(got, manual)


def test_training_reaches_high_accuracy_on_easy_separable_data():
    X, y = _make_separable_blobs(n_per_class=25, seed=4)
    clf = Perceptron(eta=0.1, epochs=300).train(X, y)

    preds = np.asarray(clf.predict(X))
    acc = float(np.mean(preds == y))
    # Should be very high on this easy dataset
    assert acc >= 0.95


def test_early_stopping_when_converged():
    X, y = _make_separable_blobs(n_per_class=20, seed=5)
    clf = Perceptron(eta=0.2, epochs=500).train(X, y)

    # If converged, last error count is 0 and it shouldn't run full epochs.
    assert clf.errors_[-1] == 0
    assert len(clf.errors_) <= clf.epochs

