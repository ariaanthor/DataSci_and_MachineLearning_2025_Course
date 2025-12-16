"""
knn.py

This module implements a simple **k-nearest neighbors** classifier using an
external distance function:

    from rice_ml.supervised_learning.util import distance

KNN is a non-parametric method: training simply stores the dataset, and
prediction classifies each query point by majority vote among the `k` closest
training samples.

Key behavior (kept identical to the original code)
--------------------------------------------------
- `fit()` stores `X_train` and `y_train` without copying or validation.
- `predict()` loops over test points one-by-one, calls `_k_nearest()`, then
  majority-votes using Python list/count operations.
- Ties are resolved by:

    max(set(votes), key=votes.count)

  Note: if there's a tie, Python's set iteration order can affect the chosen
  label. This is intentional because we are **not changing functionality**.

Examples
--------
A tiny 2D example (labels are strings):

>>> import numpy as np
>>> from rice_ml.supervised_learning.util import distance
>>> X_train = np.array([[0.0, 0.0],
...                     [0.0, 1.0],
...                     [1.0, 0.0],
...                     [1.0, 1.0]])
>>> y_train = np.array(["A", "A", "B", "B"], dtype=object)
>>> knn = KNN(k=3)
>>> knn.fit(X_train, y_train)
>>> X_test = np.array([[0.1, 0.2],
...                    [0.9, 0.8]])
>>> preds = knn.predict(X_test)
>>> len(preds)
2

Inspect the neighbors for a single point (useful for debugging):

>>> q = np.array([0.1, 0.2])
>>> neigh = knn._k_nearest(q)
>>> len(neigh) == 3
True
>>> # Each neighbor is stored as a tuple: (x_train, label, distance)
>>> isinstance(neigh[0], tuple) and len(neigh[0]) == 3
True
"""

from __future__ import annotations

# NOTE: This import is part of the original design.
# The `distance` function is assumed to take two vectors/points and return a
# scalar distance (smaller means "closer").
from rice_ml.supervised_learning.util import distance


class KNN:
    """
    K-Nearest Neighbors classifier.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider when voting.

    Attributes
    ----------
    k : int
        Stored number of neighbors.
    X_train : array_like
        Training feature matrix stored after `fit`.
    y_train : array_like
        Training labels stored after `fit`.

    Notes
    -----
    - This is a basic KNN implementation intended for clarity and small inputs.
    - The `distance` function is imported externally. That makes it easy to swap
      distance metrics (Euclidean, Manhattan, cosine, etc.) without changing this
      file.
    - No input validation is performed (kept consistent with original code).
      If you pass mismatched shapes or non-iterables, you'll see runtime errors.

    Tie-breaking
    ------------
    Prediction uses:

        pred = max(set(votes), key=votes.count)

    If multiple labels have the same highest count, the selected label may depend
    on Python's set iteration order. This is not "fixed" here because we are
    preserving the original behavior.

    Examples
    --------
    Basic usage:

    >>> import numpy as np
    >>> X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y_train = np.array([0, 0, 1, 1])
    >>> knn = KNN(k=1)
    >>> knn.fit(X_train, y_train)
    >>> knn.predict(np.array([[2.2], [0.2]]))
    [1, 0]

    Using k=3 (majority vote):

    >>> knn = KNN(k=3)
    >>> knn.fit(X_train, y_train)
    >>> knn.predict(np.array([[1.6]]))  # neighbors around 1.6 lean toward label 1
    [1]
    """

    def __init__(self, k):
        """
        Create a KNN classifier.

        Parameters
        ----------
        k : int
            Number of nearest neighbors to use.

        Notes
        -----
        This constructor stores `k` directly. It does not enforce constraints
        (e.g., k >= 1). That matches the original behavior.

        Examples
        --------
        >>> m = KNN(5)
        >>> m.k
        5
        """
        self.k = k

    def fit(self, X_train, y_train):
        """
        Store the training data.

        Parameters
        ----------
        X_train : array_like
            Training samples (e.g., NumPy array of shape (n_samples, n_features)).
        y_train : array_like
            Training labels (length n_samples).

        Returns
        -------
        None
            This method sets internal attributes and returns nothing
            (kept consistent with original code).

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> y = np.array([0, 1])
        >>> knn = KNN(k=1)
        >>> knn.fit(X, y)
        >>> knn.X_train.shape
        (2, 2)
        """
        # Training for KNN is just memorization.
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict labels for a batch of query points.

        For each test point:
        1) Find the k nearest training samples (via `_k_nearest`).
        2) Extract their labels.
        3) Choose the most common label (majority vote).

        Parameters
        ----------
        X_test : array_like
            Test samples iterable. Commonly a NumPy array of shape
            (n_test, n_features).

        Returns
        -------
        list
            A Python list of predicted labels, one per test point.

        Notes
        -----
        - This returns a list (not a NumPy array), matching the original code.
        - Runtime is O(n_test * n_train * cost(distance)).

        Examples
        --------
        >>> import numpy as np
        >>> X_train = np.array([[0.0], [1.0], [2.0]])
        >>> y_train = np.array(["L", "L", "R"], dtype=object)
        >>> knn = KNN(k=1)
        >>> knn.fit(X_train, y_train)
        >>> knn.predict(np.array([[1.8], [0.2]]))
        ['R', 'L']
        """
        predictions = []

        # Loop through each query point independently
        for point in X_test:
            # Compute the k-nearest neighbor tuples
            neighbors = self._k_nearest(point)

            # Each neighbor tuple is: (x_train, label, distance)
            votes = [label for _, label, _ in neighbors]

            # Majority vote; tie-breaking depends on set ordering (by design)
            pred = max(set(votes), key=votes.count)
            predictions.append(pred)

        return predictions

    def _k_nearest(self, point):
        """
        Compute the `k` nearest neighbors of a single query point.

        Parameters
        ----------
        point : array_like
            A single test sample (e.g., shape (n_features,)).

        Returns
        -------
        list of tuple
            A list of the `k` nearest neighbors sorted by ascending distance.
            Each element is a 3-tuple:

                (x_train, label, d)

            where:
            - x_train: the training sample
            - label:   the training label
            - d:       the computed distance from `point` to x_train

        Notes
        -----
        - This method computes distances to *all* training points, then sorts.
        - Complexity is dominated by sorting: O(n_train log n_train).
        - We keep the tuple structure and ordering exactly as originally written.

        Examples
        --------
        >>> import numpy as np
        >>> X_train = np.array([[0.0, 0.0],
        ...                     [2.0, 0.0],
        ...                     [0.0, 2.0]])
        >>> y_train = np.array([0, 1, 1])
        >>> knn = KNN(k=2)
        >>> knn.fit(X_train, y_train)
        >>> neigh = knn._k_nearest(np.array([0.1, 0.1]))
        >>> len(neigh)
        2
        >>> # Distances are sorted ascending:
        >>> neigh[0][-1] <= neigh[1][-1]
        True
        """
        neighbors = []

        # Compute distances from `point` to every training sample
        for x_train, label in zip(self.X_train, self.y_train):
            d = distance(point, x_train)
            neighbors.append((x_train, label, d))

        # Sort by the final tuple entry (distance)
        neighbors.sort(key=lambda x: x[-1])

        # Return the closest k
        return neighbors[: self.k]


# -----------------------------------------------------------------------------
# End of module
# -----------------------------------------------------------------------------

