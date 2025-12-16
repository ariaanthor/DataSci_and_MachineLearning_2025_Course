"""
Pytest unit tests for the simple KNN implementation.

"""

from __future__ import annotations

import numpy as np
import pytest

import rice_ml.supervised_learning.knn as knn_module
from rice_ml.supervised_learning.knn import KNN


def test_fit_stores_training_arrays():
    X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0, 1])

    model = KNN(k=1)
    model.fit(X_train, y_train)

    assert hasattr(model, "X_train")
    assert hasattr(model, "y_train")
    assert np.all(model.X_train == X_train)
    assert np.all(model.y_train == y_train)


def test_k_nearest_returns_sorted_neighbors_and_respects_k():
    X_train = np.array([[0.0, 0.0], [0.0, 2.0], [3.0, 0.0], [10.0, 10.0]])
    y_train = np.array([0, 0, 0, 1])

    model = KNN(k=3)
    model.fit(X_train, y_train)

    neighbors = model._k_nearest(np.array([0.0, 1.0]))

    assert isinstance(neighbors, list)
    assert len(neighbors) == 3
    # sorted by distance ascending
    dists = [t[-1] for t in neighbors]
    assert dists == sorted(dists)


def test_predict_separates_two_clusters_cleanly():
    # class 0 near origin; class 1 near (10, 10)
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [10.0, 10.0],
            [10.0, 11.0],
            [11.0, 10.0],
        ],
        dtype=float,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([[0.2, 0.1], [10.2, 10.1]], dtype=float)

    model = KNN(k=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert isinstance(preds, list)
    assert preds == [0, 1]


def test_predict_length_matches_input_length():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1])
    X_test = np.array([[0.1], [1.5], [10.0]])

    model = KNN(k=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == X_test.shape[0]


def test_distance_function_is_used(monkeypatch):
    # Verify that knn_module.distance gets called during predict.
    calls = {"n": 0}

    def fake_distance(a, b):
        calls["n"] += 1
        # simple L2
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return float(np.sqrt(np.sum((a - b) ** 2)))

    monkeypatch.setattr(knn_module, "distance", fake_distance)

    X_train = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
    y_train = np.array([0, 0, 1])
    X_test = np.array([[1.0, 0.0], [9.0, 9.0]])

    model = KNN(k=1)
    model.fit(X_train, y_train)
    _ = model.predict(X_test)

    # For each test point, distance should be computed to each training sample.
    assert calls["n"] == X_test.shape[0] * X_train.shape[0]
