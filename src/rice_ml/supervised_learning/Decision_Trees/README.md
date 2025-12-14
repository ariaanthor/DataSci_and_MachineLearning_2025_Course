# Decision Tree Classifier

This module provides a **from-scratch implementation of a CART-style decision tree classifier**. It is designed primarily for **educational use**, making the internal mechanics of decision trees—such as recursive splitting, Gini impurity, and tree traversal—easy to inspect and understand.

Unlike production libraries (e.g., scikit-learn), this implementation favors **clarity over optimization** and exposes the full learning process in readable Python code.

---

## Features

- Binary decision tree using **Gini impurity**
- Recursive tree construction
- Configurable hyperparameters:
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features` (int, float, or `None`)
- Optional `random_state` for reproducible feature subsampling
- Supports:
  - `fit(X, y)`
  - `predict(X)`
  - `predict_proba(X)`

---

## Intended Use

This code is ideal for:
- Learning how decision trees work internally
- Coursework or teaching demonstrations
- Debugging or experimenting with custom tree logic
- Comparing against library implementations

It is **not** optimized for large datasets or production deployment.

---

## Example

```python
import numpy as np
from rice_ml.supervised_learning.decision_tree import DecisionTreeClassifier

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 1, 1])

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X, y)

predictions = model.predict(X)
print(predictions)
