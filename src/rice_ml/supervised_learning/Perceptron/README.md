# Perceptron Classifier

This module implements a classic **single-layer perceptron** for **binary classification**. It is designed for clarity and instructional use rather than maximum performance, making it well-suited for coursework, experimentation, and understanding the fundamentals of linear classifiers.

## Features

* Binary classification with arbitrary label types (strings, integers, etc.)
* Online learning via the standard perceptron update rule
* Optional bias (intercept) term
* Epoch-wise shuffling with a configurable random seed
* Simple early stopping based on zero error or minimal improvement
* Scikit-learn–style API: `fit`, `predict`, `decision_function`, `score`

## Typical Workflow

1. Initialize the model with learning rate and iteration limits.
2. Call `fit(X, y)` to train on labeled data.
3. Use `predict(X)` for class labels or `decision_function(X)` for raw scores.
4. Evaluate accuracy with `score(X, y)`.

## Example

```python
import numpy as np
from rice_ml.supervised_learning.perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1])  # AND gate

model = Perceptron(learning_rate=1.0, max_iter=100)
model.fit(X, y)

print(model.predict([[1, 1]]))  # -> [1]
print(model.score(X, y))        # accuracy
```

## Notes

* The perceptron converges only when the data are linearly separable.
* Internally, labels are mapped to {-1, +1} but returned in their original form.
* This implementation prioritizes readability and robustness over speed.

This module is ideal as a foundation for understanding more advanced models such as logistic regression and multilayer neural networks.
