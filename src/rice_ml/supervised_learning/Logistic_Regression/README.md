# Logistic Regression

This module implements **logistic regression from scratch**, supporting both **binary** and **multiclass** classification. Training is performed with **batch gradient descent**, making the learning dynamics easy to inspect and modify.

---

## Features

- Binary logistic regression (sigmoid)
- Multiclass classification via:
  - **Softmax / multinomial logistic regression**
  - **One-vs-Rest (OvR)**
- Optional **L2 regularization**
- Clean, familiar API:
  - `fit`
  - `predict`
  - `predict_proba`
  - `score`

---

## Multiclass Behavior

- `multi_class="auto"` (default):
  - 2 classes → binary sigmoid model
  - >2 classes → multinomial softmax model
- `multi_class="multinomial"`:
  - Single softmax model for all classes
- `multi_class="ovr"`:
  - One binary classifier per class

---

## Quick Example

```python
import numpy as np
from rice_ml.supervised_learning.logistic_regression import LogisticRegression

X = np.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]])
y = np.array([0, 1, 1, 1])

clf = LogisticRegression(
    learning_rate=0.2,
    max_iter=2000,
    penalty="l2",
    C=1.0
).fit(X, y)

print(clf.predict([[0., 0.], [1., 1.]]))
print(clf.predict_proba([[0., 0.], [1., 1.]]))
