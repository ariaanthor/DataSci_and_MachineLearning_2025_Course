"""
ensemble_methods.py

Collection of simple ensemble learners built on top of the package's
from-scratch decision tree classifier.

What lives here?
- RandomForestClassifier: many trees + bootstrap rows + random feature subsets
- BaggingClassifier: bootstrap aggregating with per-estimator feature subsampling
- AdaBoostClassifier: SAMME-style boosting with decision stumps (depth=1 trees)

These implementations are intentionally "plain" (NumPy only) so the training
loops are readable and easy to step through in a debugger.

Quick sanity check
------------------
>>> import numpy as np
>>> from rice_ml.supervised_learning.ensemble_methods import AdaBoostClassifier
>>> X = np.array([[0., 0.],
...               [0., 1.],
...               [1., 0.],
...               [1., 1.]])
>>> y = np.array([0, 1, 1, 0])
>>> model = AdaBoostClassifier(n_estimators=5, random_state=1)
>>> model.fit(X, y).predict([[0., 0.], [1., 1.]])
array([0, 0])
"""

from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np

from ..Decision_Trees.decision_trees import DecisionTreeClassifier

__all__ = [
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "BaggingClassifier",
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Normalize an input into a 2D float NumPy array.

    Rules
    -----
    - 1D inputs are interpreted as a single feature and reshaped to (n, 1).
    - Only 1D/2D inputs are accepted.
    - Empty inputs are rejected.
    - Non-numeric dtypes are coerced to float when possible.

    Parameters
    ----------
    X : ArrayLike
        Candidate data array.
    name : str
        Used to produce friendlier error messages.

    Returns
    -------
    np.ndarray
        A float array with ndim==2.

    Raises
    ------
    ValueError
        If `X` is not 1D or 2D, or is empty.
    TypeError
        If conversion to float fails.
    """
    arr = np.asarray(X)

    # Allow callers to pass a single feature vector directly
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Everything in this module assumes (n_samples, n_features)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got {arr.ndim}D.")

    # A surprisingly common failure mode when users slice incorrectly
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    # Coerce to float; if elements aren't numeric, this should fail loudly
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)

    return arr


def _ensure_1d(y: ArrayLike, name: str = "y") -> np.ndarray:
    """
    Ensure a target array is 1D and non-empty.

    Parameters
    ----------
    y : ArrayLike
        Candidate label vector.
    name : str
        Used only for error strings.

    Returns
    -------
    np.ndarray
        1D array view/copy of the input.

    Raises
    ------
    ValueError
        If `y` is not 1D or has zero length.
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


class RandomForestClassifier:
    """
    Random forest classifier (bagged trees + random feature selection).

    Training procedure (high-level)
    -------------------------------
    For each tree:
      1) draw a bootstrap sample of rows (if bootstrap=True)
      2) train a DecisionTreeClassifier with a restricted feature set size
         (controlled by `max_features`)

    Prediction aggregates by averaging class probabilities from all trees
    and taking argmax.

    Example
    -------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.ensemble_methods import RandomForestClassifier
    >>> X = np.array([[0., 0.],
    ...               [0., 1.],
    ...               [1., 0.],
    ...               [1., 1.]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rf = RandomForestClassifier(n_estimators=3, random_state=7)
    >>> rf.fit(X, y).predict([[0., 0.], [1., 1.]])
    array([0, 1])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float, None] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        # Basic parameter sanity checks
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        # Populated by fit(...)
        self.estimators_: list = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def _get_max_features(self, n_features: int) -> int:
        """
        Translate `max_features` into a concrete integer count.

        Accepted forms:
        - None: use all features
        - "sqrt": floor(sqrt(n_features)), at least 1
        - "log2": floor(log2(n_features)), at least 1
        - int: clipped to [1, n_features]
        - float: fraction of n_features, at least 1

        Returns
        -------
        int
            Number of features to consider per split.
        """
        if self.max_features is None:
            return n_features
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RandomForestClassifier":
        """
        Build the forest from training data.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix of shape (n_samples, n_features).
        y : ArrayLike
            Integer class labels of shape (n_samples,). Labels must be encoded
            as 0..K-1 (the underlying tree assumes this).

        Returns
        -------
        self : RandomForestClassifier
            Fitted model.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        if not np.issubdtype(y_arr.dtype, np.integer):
            raise ValueError("y must be integer-encoded (0, 1, 2, ...).")

        # Cache label metadata
        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)

        n_samples, n_features = X_arr.shape
        max_features = self._get_max_features(n_features)

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            # Choose rows for this tree (bootstrap by default)
            if self.bootstrap:
                indices = rng.integers(0, n_samples, size=n_samples)
            else:
                indices = np.arange(n_samples)

            X_boot = X_arr[indices]
            y_boot = y_arr[indices]

            # Each estimator gets its own RNG seed for split-time feature subsampling
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=rng.integers(0, 2**31),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Estimate class probabilities by averaging over trees.

        Parameters
        ----------
        X : ArrayLike
            Samples shaped (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array shaped (n_samples, n_classes_) containing mean predicted
            probabilities across all estimators.

        Raises
        ------
        RuntimeError
            If `fit` has not been called.
        """
        if not self.estimators_:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")

        # Stack: (n_trees, n_samples, n_classes) then average over first axis
        all_proba = np.array([tree.predict_proba(X_arr) for tree in self.estimators_])
        return np.mean(all_proba, axis=0)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class indices (argmax over predicted probabilities).

        Parameters
        ----------
        X : ArrayLike
            Input samples.

        Returns
        -------
        np.ndarray
            Integer class predictions.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute simple classification accuracy.
        """
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))


class BaggingClassifier:
    """
    Bagging classifier with feature subsampling per estimator.

    Each base estimator is trained on:
      - a (potentially bootstrapped) subset of rows, and
      - a randomly selected subset of columns

    Final prediction uses majority vote over estimator outputs.

    Example
    -------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.ensemble_methods import BaggingClassifier
    >>> X = np.array([[1., 0.],
    ...               [0., 1.],
    ...               [1., 1.],
    ...               [0., 0.]])
    >>> y = np.array([1, 1, 0, 0])
    >>> bag = BaggingClassifier(n_estimators=3, random_state=2)
    >>> bag.fit(X, y).predict([[1., 0.], [0., 0.]])
    array([1, 0])
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if not 0 < max_samples <= 1.0:
            raise ValueError("max_samples must be in (0, 1].")
        if not 0 < max_features <= 1.0:
            raise ValueError("max_features must be in (0, 1].")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: list = []
        self.estimator_features_: list = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "BaggingClassifier":
        """
        Train each base estimator on its own subsample.

        Parameters
        ----------
        X : ArrayLike
            Training features.
        y : ArrayLike
            Integer-coded training labels.

        Returns
        -------
        self : BaggingClassifier
            Trained model.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        if not np.issubdtype(y_arr.dtype, np.integer):
            raise ValueError("y must be integer-encoded.")

        self.classes_ = np.unique(y_arr)
        n_samples, n_features = X_arr.shape

        n_samples_draw = max(1, int(self.max_samples * n_samples))
        n_features_draw = max(1, int(self.max_features * n_features))

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.estimator_features_ = []

        for _ in range(self.n_estimators):
            # Row indices: bootstrap (with replacement) or subsample (without)
            if self.bootstrap:
                sample_idx = rng.integers(0, n_samples, size=n_samples_draw)
            else:
                sample_idx = rng.choice(n_samples, size=n_samples_draw, replace=False)

            # Column indices: always without replacement
            feature_idx = rng.choice(n_features, size=n_features_draw, replace=False)
            feature_idx = np.sort(feature_idx)  # stable ordering (debug-friendly)

            X_subset = X_arr[sample_idx][:, feature_idx]
            y_subset = y_arr[sample_idx]

            # Use a tree as the default "base estimator"
            tree = DecisionTreeClassifier(random_state=rng.integers(0, 2**31))
            tree.fit(X_subset, y_subset)

            self.estimators_.append(tree)
            self.estimator_features_.append(feature_idx)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict by majority vote over the ensemble.

        Parameters
        ----------
        X : ArrayLike
            Samples to classify.

        Returns
        -------
        np.ndarray
            Predicted class label for each sample.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        if not self.estimators_:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        n_samples = X_arr.shape[0]

        # votes[i, c] counts how many estimators predicted class c for sample i
        votes = np.zeros((n_samples, len(self.classes_)), dtype=int)

        for tree, features in zip(self.estimators_, self.estimator_features_):
            preds = tree.predict(X_arr[:, features])
            for i, p in enumerate(preds):
                votes[i, p] += 1

        return np.argmax(votes, axis=1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute accuracy on labeled data.
        """
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))


class AdaBoostClassifier:
    """
    AdaBoost classifier using decision stumps and the SAMME update.

    Each round fits a weak learner on a weighted sample of the training set.
    Learners are combined via a weighted vote, with weights derived from the
    learner's weighted error.

    Example
    -------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.ensemble_methods import AdaBoostClassifier
    >>> X = np.array([[1., 1.],
    ...               [1., 0.],
    ...               [0., 1.],
    ...               [0., 0.]])
    >>> y = np.array([1, 1, 0, 0])
    >>> ada = AdaBoostClassifier(n_estimators=4, random_state=3)
    >>> ada.fit(X, y).predict([[1., 1.], [0., 0.]])
    array([1, 0])
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.estimators_: list = []
        self.estimator_weights_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AdaBoostClassifier":
        """
        Train an AdaBoost ensemble.

        Parameters
        ----------
        X : ArrayLike
            Training features.
        y : ArrayLike
            Integer-coded labels.

        Returns
        -------
        self : AdaBoostClassifier
            Fitted booster.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        if not np.issubdtype(y_arr.dtype, np.integer):
            raise ValueError("y must be integer-encoded.")

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        n_samples = X_arr.shape[0]

        # Start with uniform weights over training samples
        sample_weights = np.ones(n_samples) / n_samples

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        estimator_weights = []

        for _ in range(self.n_estimators):
            # Weak learner: depth-1 tree (a stump)
            tree = DecisionTreeClassifier(
                max_depth=1,
                random_state=rng.integers(0, 2**31),
            )

            # Weighted bootstrap according to sample_weights
            indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights,
            )
            tree.fit(X_arr[indices], y_arr[indices])

            # Evaluate on the full training set
            y_pred = tree.predict(X_arr)
            incorrect = (y_pred != y_arr).astype(float)

            # Weighted classification error
            error = np.sum(sample_weights * incorrect)

            # Prevent numerical explosions when error hits 0 or 1
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # SAMME weight for multiclass boosting
            alpha = self.learning_rate * (
                np.log((1 - error) / error) + np.log(n_classes - 1)
            )

            # Reweight samples: misclassified points get upweighted
            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= np.sum(sample_weights)

            self.estimators_.append(tree)
            estimator_weights.append(alpha)

        self.estimator_weights_ = np.array(estimator_weights)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict by weighted voting of weak learners.
        """
        if not self.estimators_:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)

        # Accumulate per-class vote weights for each sample
        class_weights = np.zeros((n_samples, n_classes))

        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X_arr)
            for i, p in enumerate(preds):
                class_weights[i, p] += alpha

        return np.argmax(class_weights, axis=1)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Produce normalized class weights (a probability-like output).

        Note: This normalization treats accumulated vote weights as unnormalized
        scores and rescales them to sum to 1 across classes per sample.
        """
        if not self.estimators_:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)

        class_weights = np.zeros((n_samples, n_classes))

        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X_arr)
            for i, p in enumerate(preds):
                class_weights[i, p] += alpha

        # Convert weights to probabilities by row-normalization
        proba = class_weights / class_weights.sum(axis=1, keepdims=True)
        return proba

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Accuracy convenience method.
        """
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))
