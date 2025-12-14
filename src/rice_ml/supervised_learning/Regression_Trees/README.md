# Decision Tree Regressor

This module implements a **CART-style decision tree for regression**. It uses **mean squared error (MSE)** as the splitting criterion and is designed to be easy to read, debug, and extend—making it ideal for educational use and small-scale experiments.

## Features

* Pure NumPy implementation (no external ML libraries)
* Recursive tree construction with greedy split selection
* Mean-squared-error–based impurity reduction
* Configurable stopping criteria:

  * `max_depth`
  * `min_samples_split`
  * `min_samples_leaf`
* Optional feature subsampling (`max_features`) for ensemble compatibility
* Scikit-learn–style API: `fit`, `predict`, `score`

## Typical Workflow

1. Initialize the regressor with desired hyperparameters.
2. Train the model using `fit(X, y)`.
3. Generate predictions with `predict(X)`.
4. Evaluate performance using `score(X, y)` (R²).

## Example

```python
import numpy as np
from rice_ml.supervised_learning.regression_trees import DecisionTreeRegressor

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

reg = DecisionTreeRegressor(max_depth=2)
reg.fit(X, y)

print(reg.predict([[2.5]]))   # predicted value
print(reg.score(X, y))        # R^2 score
```

## Notes

* Leaf nodes store the **mean target value** of samples reaching that node.
* A split is only accepted if it reduces variance compared to the parent node.
* The implementation prioritizes clarity and correctness over raw performance.

This regressor serves as a strong foundation for understanding tree-based models and can be extended to ensembles such as Random Forests or Gradient Boosting.
