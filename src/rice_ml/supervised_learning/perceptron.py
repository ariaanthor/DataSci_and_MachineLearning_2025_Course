"""
perceptron.py

This module implements the classic single-layer Perceptron learning algorithm
for **binary classification** using a hard threshold (step) activation.

The perceptron learns a linear decision boundary by iteratively updating a
weight vector whenever it misclassifies a training example.

Important notes
---------------
- This implementation expects labels in {-1, +1}.
- The update rule used here is:

    update = eta * (target - prediction)

  where prediction is either -1 or +1 from the current model.

- Training stops early if an entire epoch completes with 0 errors.

This file is intentionally "verbose" (lots of comments + docstrings) to be
educational and test-friendly, while preserving the original functionality.

Examples
--------
Train on a simple AND-like dataset (with labels -1/+1):

>>> import numpy as np
>>> X = np.array([[0.0, 0.0],
...               [0.0, 1.0],
...               [1.0, 0.0],
...               [1.0, 1.0]])
>>> y = np.array([-1, -1, -1,  1])  # AND (only [1,1] is positive)
>>> p = Perceptron(eta=0.5, epochs=50).train(X, y)
>>> p.predict(np.array([1.0, 1.0]))
array(1)
>>> p.predict(np.array([0.0, 1.0]))
array(-1)

Check that training converged early (often, but not guaranteed for every random init):

>>> len(p.errors_) <= p.epochs
True

"""

from __future__ import annotations

import numpy as np


class Perceptron(object):
    """
    Perceptron binary classifier.

    The perceptron is one of the simplest neural models: a linear classifier
    that outputs either -1 or +1 depending on the sign of a linear score.

    Parameters
    ----------
    eta : float, default=0.5
        Learning rate (step size) for weight updates.
        Higher values update more aggressively; smaller values update more slowly.

    epochs : int, default=50
        Maximum number of passes over the training data.

    Attributes
    ----------
    w_ : np.ndarray of shape (n_features + 1,)
        Weight vector learned during training.
        The last element is treated as the bias/intercept term.

        - w_[:-1] are feature weights
        - w_[-1]  is the bias

    errors_ : list of int
        Number of updates (misclassifications) made in each epoch.

    Notes
    -----
    Labels are expected to be in {-1, +1}. If your data uses {0, 1}, convert:

    >>> y01 = np.array([0, 1, 0, 1])
    >>> ypm = np.where(y01 == 1, 1, -1)
    >>> ypm.tolist()
    [-1, 1, -1, 1]

    The prediction uses a step function at 0:

    - net_input(x) >= 0 -> +1
    - net_input(x) <  0 -> -1

    Examples
    --------
    A tiny linearly separable dataset:

    >>> import numpy as np
    >>> X = np.array([[ 2.0,  1.0],
    ...               [ 3.0,  1.0],
    ...               [-2.0, -1.0],
    ...               [-3.0, -2.0]])
    >>> y = np.array([1, 1, -1, -1])
    >>> clf = Perceptron(eta=0.2, epochs=30).train(X, y)
    >>> preds = clf.predict(X)
    >>> (preds == y).all()
    np.True_

    Non-separable example (XOR) will not converge perfectly:

    >>> X = np.array([[0.0, 0.0],
    ...               [0.0, 1.0],
    ...               [1.0, 0.0],
    ...               [1.0, 1.0]])
    >>> y = np.array([-1, 1, 1, -1])  # XOR in -1/+1
    >>> clf = Perceptron(eta=0.5, epochs=20).train(X, y)
    >>> len(clf.errors_) <= 20
    True
    """

    def __init__(self, eta: float = 0.5, epochs: int = 50):
        """
        Initialize a Perceptron instance.

        Parameters
        ----------
        eta : float, default=0.5
            Learning rate for weight updates.

        epochs : int, default=50
            Maximum number of training epochs.

        Notes
        -----
        This constructor only stores hyperparameters. Weights are initialized
        when `train()` is called, because we need to know the feature dimension.

        Examples
        --------
        >>> p = Perceptron(eta=0.1, epochs=10)
        >>> p.eta, p.epochs
        (0.1, 10)
        """
        self.eta = eta
        self.epochs = epochs

        # The following attributes are created during training:
        #   - self.w_       (weights + bias)
        #   - self.errors_  (epoch-wise misclassification counts)

    def train(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Fit the perceptron model on training data.

        This method implements the standard perceptron training loop:
        repeatedly iterate through samples, updating weights when a sample
        is misclassified.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix where each row is a sample.

        y : np.ndarray of shape (n_samples,)
            Target labels for each sample, expected to be -1 or +1.

        Returns
        -------
        self : Perceptron
            Returns the fitted instance (enables chaining).

        How training works
        ------------------
        1. Initialize weights randomly for each feature, plus a bias.
        2. For each epoch:
           - iterate over each (xi, target)
           - compute prediction = self.predict(xi)
           - compute update = eta * (target - prediction)
           - apply update to weights and bias
           - count whether an update occurred (misclassification)
        3. Stop early if an epoch finishes with 0 updates.

        Examples
        --------
        Basic usage:

        >>> import numpy as np
        >>> X = np.array([[1.0, 0.0],
        ...               [0.0, 1.0],
        ...               [1.0, 1.0]])
        >>> y = np.array([-1, -1,  1])
        >>> model = Perceptron(eta=0.5, epochs=50).train(X, y)
        >>> model.w_.shape
        (3,)

        Training returns `self`:

        >>> isinstance(model, Perceptron)
        True
        """
        # --- Weight initialization ------------------------------------------------
        # We allocate one weight per feature, plus one bias term.
        #
        #   w_[:-1] -> feature weights
        #   w_[-1]  -> bias
        #
        # Random initialization matches the original code's behavior.
        self.w_ = np.random.rand(1 + X.shape[1])

        # Track the number of updates (misclassifications) each epoch
        self.errors_ = []

        # --- Main training loop ---------------------------------------------------
        for _ in range(self.epochs):
            errors = 0

            # Iterate through samples one by one (online / stochastic style)
            for xi, target in zip(X, y):
                # Predict current class for this sample
                pred = self.predict(xi)

                # Compute update:
                #
                # If pred == target, then (target - pred) == 0 -> no update
                # If pred != target, then (target - pred) is ±2 -> update occurs
                update = self.eta * (target - pred)

                # Update feature weights
                self.w_[:-1] += update * xi

                # Update bias/intercept
                self.w_[-1] += update

                # Count if an update happened (i.e., sample was misclassified)
                errors += int(update != 0)

            # Store epoch error count
            self.errors_.append(errors)

            # Early stopping: perfect classification on this pass
            if errors == 0:
                break

        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear score (net input) for a sample or batch.

        The net input is:

            score = X · w + b

        where:
        - w is `self.w_[:-1]`
        - b is `self.w_[-1]`

        Parameters
        ----------
        X : np.ndarray
            Either:
            - shape (n_features,) for a single sample, or
            - shape (n_samples, n_features) for a batch.

        Returns
        -------
        np.ndarray
            The linear score(s). For a single sample, this will be a scalar-like
            NumPy value; for a batch, it will be a 1D array of scores.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[1.0, 2.0],
        ...               [0.0, 0.0]])
        >>> y = np.array([1, -1])
        >>> p = Perceptron(eta=0.5, epochs=10).train(X, y)
        >>> s = p.net_input(np.array([1.0, 2.0]))
        >>> np.isfinite(s).all()
        np.True_
        """
        # np.dot handles both vector and matrix input appropriately here:
        # - (n_features,) dot (n_features,) -> scalar
        # - (n_samples, n_features) dot (n_features,) -> (n_samples,)
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class label(s) for input sample(s).

        This uses a threshold at 0 on the net input:
        - net_input >= 0 -> +1
        - net_input <  0 -> -1

        Parameters
        ----------
        X : np.ndarray
            Either:
            - shape (n_features,) for a single sample, or
            - shape (n_samples, n_features) for a batch.

        Returns
        -------
        np.ndarray
            Predicted labels in {-1, +1}.
            For a single sample input, NumPy returns a scalar-like array value.

        Examples
        --------
        Single sample:

        >>> import numpy as np
        >>> X = np.array([[1.0], [2.0], [3.0]])
        >>> y = np.array([-1, -1, 1])
        >>> p = Perceptron(eta=0.5, epochs=20).train(X, y)
        >>> p.predict(np.array([3.0])) in (-1, 1)
        True

        Batch prediction:

        >>> preds = p.predict(X)
        >>> preds.shape
        (3,)
        >>> set(preds.tolist()).issubset({-1, 1})
        True
        """
        # We deliberately keep the original "hard threshold" behavior.
        #
        # np.where works for scalars and arrays:
        # - if net_input returns a scalar -> returns a scalar-like array value
        # - if net_input returns a vector -> returns a vector of labels
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# -----------------------------------------------------------------------------
# End of module
# -----------------------------------------------------------------------------