"""
multilayer_perceptron.py

A small Multilayer Perceptron (MLP).

What this file provides
-----------------------
- `MLPClassifier`: feedforward neural net for classification
- `MLPRegressor`: feedforward neural net for regression

Training uses vanilla backpropagation with (optional) mini-batches and a simple
momentum term. The goal is readability and a minimal dependency footprint.

Notes
-----
- Hidden layers share a single activation choice: {"relu", "sigmoid", "tanh"}.
- For classification:
  * binary -> sigmoid output with 1 unit, probabilities returned as (n, 2)
  * multiclass -> softmax output with K units
- For regression:
  * identity output with 1 unit

Example (classifier)
--------------------
>>> import numpy as np
>>> from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier
>>> X = np.array([[1., 0.],
...               [0., 1.],
...               [1., 1.],
...               [0., 0.]])
>>> y = np.array(["yes", "yes", "no", "no"], dtype=object)
>>> net = MLPClassifier(hidden_layer_sizes=(6,), max_iter=200, random_state=0).fit(X, y)
>>> net.predict([[1., 0.], [0., 0.]]).tolist()
['yes', 'no']

Example (regressor)
-------------------
>>> from rice_ml.supervised_learning.multilayer_perceptron import MLPRegressor
>>> X = np.array([[0.], [1.], [2.]], dtype=float)
>>> y = np.array([0., 1., 4.], dtype=float)
>>> reg = MLPRegressor(hidden_layer_sizes=(5,), max_iter=50, random_state=1).fit(X, y)
>>> reg.predict([[1.]])[0] >= 0
True
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple, Union, Sequence

import numpy as np

__all__ = ["MLPClassifier", "MLPRegressor"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# -----------------------------------------------------------------------------
# Input utilities
# -----------------------------------------------------------------------------

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Coerce X to a 2D float ndarray.

    Behavior
    --------
    - If X is 1D, it is treated as a single feature and reshaped to (n, 1).
    - Non-numeric dtypes are cast to float (or raise).
    - Empty inputs are rejected.
    """
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
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
    Coerce y to a 1D ndarray and validate non-empty.

    For classification, labels can be strings or ints.
    For regression, the regressor converts to float later.
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


# -----------------------------------------------------------------------------
# Activations
# -----------------------------------------------------------------------------

def _relu(z: np.ndarray) -> np.ndarray:
    """ReLU nonlinearity: max(0, z)."""
    return np.maximum(0, z)


def _relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU w.r.t z (0 at z<=0, 1 at z>0)."""
    return (z > 0).astype(float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid with clipping to avoid exp overflow."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid using s*(1-s)."""
    s = _sigmoid(z)
    return s * (1 - s)


def _tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent."""
    return np.tanh(z)


def _tanh_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of tanh."""
    return 1 - np.tanh(z) ** 2


def _softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax over axis=1.

    Uses a max-shift for improved numerical stability.
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _identity(z: np.ndarray) -> np.ndarray:
    """Identity (used for regression output)."""
    return z


def _identity_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of identity."""
    return np.ones_like(z)


# Map activation names to (function, derivative)
ACTIVATIONS = {
    "relu": (_relu, _relu_derivative),
    "sigmoid": (_sigmoid, _sigmoid_derivative),
    "tanh": (_tanh, _tanh_derivative),
    "identity": (_identity, _identity_derivative),
}


class _BaseMLP:
    """
    Shared scaffolding for MLPClassifier and MLPRegressor.

    This base class owns:
    - hyperparameter validation
    - parameter initialization
    - forward pass through hidden layers
    - bookkeeping: n_iter_, loss_curve_

    Subclasses specify:
    - output activation
    - loss function
    - output gradient definition
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: Literal["relu", "sigmoid", "tanh"] = "relu",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
        momentum: float = 0.9,
    ) -> None:
        if not hidden_layer_sizes or not all(h > 0 for h in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must be a tuple of positive integers.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS.keys())}.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.batch_size = batch_size
        self.momentum = momentum

        # Parameters (lists indexed by layer transition)
        self.weights_: Optional[list] = None
        self.biases_: Optional[list] = None

        # Training metadata
        self.n_iter_: int = 0
        self.loss_curve_: list = []

    def _initialize_weights(self, layer_sizes: list, rng: np.random.Generator) -> None:
        """
        Xavier/Glorot-style initialization for each layer connection.

        layer_sizes is a list like [n_in, h1, h2, ..., n_out].
        For each i: W_i shape (layer_sizes[i], layer_sizes[i+1]).
        """
        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            # Keep variance roughly controlled as depth grows
            std = np.sqrt(2.0 / (fan_in + fan_out))

            W = rng.normal(0, std, (fan_in, fan_out))
            b = np.zeros(fan_out)

            self.weights_.append(W)
            self.biases_.append(b)

    def _forward(self, X: np.ndarray) -> Tuple[list, list]:
        """
        Forward propagate X through the network.

        Returns
        -------
        activations : list
            activations[0] = X
            activations[i+1] = activation after layer i
        z_values : list
            Pre-activation values for each layer.

        Notes
        -----
        - Hidden layers use self.activation
        - Output layer uses the subclass's _output_activation
        """
        activations = [X]
        z_values = []

        act_fn, _ = ACTIVATIONS[self.activation]

        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = activations[-1] @ W + b
            z_values.append(z)

            if i < len(self.weights_) - 1:
                # hidden layer transform
                a = act_fn(z)
            else:
                # output layer transform
                a = self._output_activation(z)

            activations.append(a)

        return activations, z_values

    def _output_activation(self, z: np.ndarray) -> np.ndarray:
        """Subclass provides output nonlinearity."""
        raise NotImplementedError

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Subclass provides the objective."""
        raise NotImplementedError

    def _output_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Subclass provides gradient of the loss w.r.t output activations."""
        raise NotImplementedError


class MLPClassifier(_BaseMLP):
    """
    Feedforward neural network classifier.

    The network uses:
    - hidden activation: relu/sigmoid/tanh
    - output: sigmoid (binary) or softmax (multiclass)

    Example
    -------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier
    >>> X = np.array([[2., 0.],
    ...               [0., 2.],
    ...               [2., 2.],
    ...               [0., 0.]])
    >>> y = np.array([1, 1, 0, 0])
    >>> clf = MLPClassifier(hidden_layer_sizes=(3,), max_iter=100, random_state=2).fit(X, y)
    >>> clf.predict([[2., 0.], [0., 0.]]).tolist()
    [1, 0]
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: Literal["relu", "sigmoid", "tanh"] = "relu",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            batch_size=batch_size,
            momentum=momentum,
        )
        self.classes_: Optional[np.ndarray] = None
        self._n_outputs: int = 0

    def _output_activation(self, z: np.ndarray) -> np.ndarray:
        # Binary uses a single sigmoid unit; multiclass uses K-way softmax
        if self._n_outputs == 1:
            return _sigmoid(z)
        return _softmax(z)

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Cross-entropy style loss.

        Note: This matches the original implementation (same formula for both
        binary and multiclass encodings).
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _output_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # For softmax+CE (and this chosen loss), gradient simplifies to y_pred - y_true
        return y_pred - y_true

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPClassifier":
        """
        Train the classifier.

        Parameters
        ----------
        X : array_like
            Feature matrix of shape (n_samples, n_features).
        y : array_like
            Labels of shape (n_samples,). Can be strings or numeric.

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("y must contain at least 2 classes.")

        n_samples, n_features = X_arr.shape

        # Encode y into the shape expected by the loss/grad pipeline
        if n_classes == 2:
            self._n_outputs = 1
            y_encoded = (y_arr == self.classes_[1]).astype(float).reshape(-1, 1)
        else:
            self._n_outputs = n_classes
            class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_idx = np.array([class_to_idx[label] for label in y_arr])
            y_encoded = np.zeros((n_samples, n_classes))
            y_encoded[np.arange(n_samples), y_idx] = 1

        # Assemble network layer sizes
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [self._n_outputs]

        rng = np.random.default_rng(self.random_state)
        self._initialize_weights(layer_sizes, rng)

        # Momentum buffers (velocity)
        v_weights = [np.zeros_like(W) for W in self.weights_]
        v_biases = [np.zeros_like(b) for b in self.biases_]

        batch_size = self.batch_size or n_samples
        _, act_derivative = ACTIVATIONS[self.activation]

        self.loss_curve_ = []

        for epoch in range(self.max_iter):
            # Shuffle each epoch for mini-batch behavior
            indices = rng.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                X_batch = X_arr[batch_idx]
                y_batch = y_encoded[batch_idx]

                # ---- Forward pass ----
                activations, z_values = self._forward(X_batch)

                # ---- Backward pass ----
                m = len(batch_idx)
                delta = self._output_gradient(y_batch, activations[-1])

                # Iterate backwards through layers
                for i in range(len(self.weights_) - 1, -1, -1):
                    grad_W = (1 / m) * activations[i].T @ delta
                    grad_b = (1 / m) * np.sum(delta, axis=0)

                    # Momentum SGD update
                    v_weights[i] = self.momentum * v_weights[i] - self.learning_rate * grad_W
                    v_biases[i] = self.momentum * v_biases[i] - self.learning_rate * grad_b

                    self.weights_[i] += v_weights[i]
                    self.biases_[i] += v_biases[i]

                    # Push delta to previous layer (if any hidden layer remains)
                    if i > 0:
                        delta = (delta @ self.weights_[i].T) * act_derivative(z_values[i - 1])

            # Track loss after epoch on full training set
            _, z_vals = self._forward(X_arr)
            y_pred = self._output_activation(z_vals[-1])
            loss = self._compute_loss(y_encoded, y_pred)
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

            # Stop if loss improvement becomes tiny
            if len(self.loss_curve_) > 1:
                if abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < self.tol:
                    break

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Probability estimates.

        Returns
        -------
        ndarray
            - binary: shape (n_samples, 2) = [P(class0), P(class1)]
            - multiclass: shape (n_samples, n_classes)
        """
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        activations, _ = self._forward(X_arr)
        output = activations[-1]

        # Convert single sigmoid output into a 2-column probability table
        if self._n_outputs == 1:
            return np.column_stack([1 - output.ravel(), output.ravel()])
        return output

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Hard label prediction (argmax over predict_proba)."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Accuracy on (X, y)."""
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))


class MLPRegressor(_BaseMLP):
    """
    Feedforward neural network regressor.

    Uses identity output and mean squared error.

    Example
    -------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.multilayer_perceptron import MLPRegressor
    >>> X = np.array([[0.], [1.], [2.], [3.]], dtype=float)
    >>> y = np.array([0., 1., 4., 9.], dtype=float)
    >>> reg = MLPRegressor(hidden_layer_sizes=(4,), max_iter=30, random_state=0).fit(X, y)
    >>> reg.predict([[2.]])[0] >= 0
    True
    """

    def _output_activation(self, z: np.ndarray) -> np.ndarray:
        # Regression output is linear
        return z

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def _output_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Matches original: scaled by 1/n (note: this is then used with batch sizes)
        return 2 * (y_pred - y_true) / y_true.shape[0]

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPRegressor":
        """
        Train the regressor.

        Parameters
        ----------
        X : array_like
            Training inputs, shape (n_samples, n_features).
        y : array_like
            Targets, shape (n_samples,). Converted to float.

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y").astype(float).reshape(-1, 1)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        n_samples, n_features = X_arr.shape

        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]

        rng = np.random.default_rng(self.random_state)
        self._initialize_weights(layer_sizes, rng)

        v_weights = [np.zeros_like(W) for W in self.weights_]
        v_biases = [np.zeros_like(b) for b in self.biases_]

        batch_size = self.batch_size or n_samples
        _, act_derivative = ACTIVATIONS[self.activation]

        self.loss_curve_ = []

        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                X_batch = X_arr[batch_idx]
                y_batch = y_arr[batch_idx]

                activations, z_values = self._forward(X_batch)

                # Backprop starts from regression output layer
                delta = self._output_gradient(y_batch, activations[-1])

                for i in range(len(self.weights_) - 1, -1, -1):
                    grad_W = activations[i].T @ delta
                    grad_b = np.sum(delta, axis=0)

                    v_weights[i] = self.momentum * v_weights[i] - self.learning_rate * grad_W
                    v_biases[i] = self.momentum * v_biases[i] - self.learning_rate * grad_b

                    self.weights_[i] += v_weights[i]
                    self.biases_[i] += v_biases[i]

                    if i > 0:
                        delta = (delta @ self.weights_[i].T) * act_derivative(z_values[i - 1])

            # Loss tracked over full dataset
            activations, _ = self._forward(X_arr)
            loss = self._compute_loss(y_arr, activations[-1])
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

            if len(self.loss_curve_) > 1:
                if abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < self.tol:
                    break

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous targets."""
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        activations, _ = self._forward(X_arr)
        return activations[-1].ravel()

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """R^2 coefficient of determination."""
        y_arr = _ensure_1d(y, "y").astype(float)
        y_pred = self.predict(X)

        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)
