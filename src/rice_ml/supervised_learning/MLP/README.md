# Multilayer Perceptron

This module implements a lightweight **feedforward neural network** (MLP), intended for learning and small-scale experiments.

## What’s included

* **`MLPClassifier`**

  * Binary classification (sigmoid output) and multiclass classification (softmax output)
  * `fit`, `predict`, `predict_proba`, `score`
* **`MLPRegressor`**

  * Regression with a linear output layer
  * `fit`, `predict`, `score`

## Training approach

* Forward pass + **backpropagation**
* Gradient descent updates with **momentum**
* Optional **mini-batches** via `batch_size`
* Early stopping based on small change in loss (`tol`)

## Key parameters (high level)

* `hidden_layer_sizes`: tuple like `(64, 32)` for two hidden layers
* `activation`: `"relu"`, `"sigmoid"`, or `"tanh"` (hidden layers)
* `learning_rate`, `max_iter`, `tol`
* `batch_size`: `None` means full-batch
* `momentum`
* `random_state`: reproducible initialization / shuffling

## Quick start

### Classification

```python
import numpy as np
from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0, 1, 1, 0])  # XOR-ish

clf = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", max_iter=2000, random_state=0)
clf.fit(X, y)

print(clf.predict(X))
print(clf.predict_proba([[0., 1.]]))
print("acc:", clf.score(X, y))
```

### Regression

```python
import numpy as np
from rice_ml.supervised_learning.multilayer_perceptron import MLPRegressor

X = np.array([[0.], [1.], [2.], [3.]])
y = np.array([0., 1., 4., 9.])

reg = MLPRegressor(hidden_layer_sizes=(10,), activation="relu", max_iter=500, random_state=1)
reg.fit(X, y)

print(reg.predict([[2.5]]))
print("r2:", reg.score(X, y))
```

## Outputs and conventions

* `MLPClassifier.predict_proba(...)`

  * Binary: returns shape `(n_samples, 2)` → `[P(class0), P(class1)]`
  * Multiclass: returns shape `(n_samples, n_classes)`
* `score`

  * Classifier: accuracy
  * Regressor: R²

## Dependencies

* Python 3.x
* NumPy
