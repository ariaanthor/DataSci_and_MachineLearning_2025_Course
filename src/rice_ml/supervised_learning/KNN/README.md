# k-Nearest Neighbors

This module contains implementations of k-Nearest Neighbors for both **classification** and **regression**. The design is intentionally simple and readable: fitting stores the training data, and prediction performs a nearest-neighbor search at query time.

---

## What’s Included

### `KNNClassifier`
- Predicts discrete labels using neighbor voting
- Supports `predict_proba()` for class probability estimates
- Voting can be:
  - `weights="uniform"`: each neighbor counts equally
  - `weights="distance"`: inverse-distance weighting (with exact-match handling)

### `KNNRegressor`
- Predicts continuous values by averaging neighbor targets
- Supports uniform or distance-weighted averaging
- Includes an `R^2`-based `score()` method with safeguards for constant targets

---

## Key Options

- `n_neighbors`: number of neighbors (k)
- `metric`: `"euclidean"` or `"manhattan"`
- `weights`: `"uniform"` or `"distance"`

---

## Quick Start

```python
import numpy as np
from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor

# Classification
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0, 0, 1, 1])
clf = KNNClassifier(n_neighbors=3, metric="euclidean").fit(X, y)
print(clf.predict([[0.2, 0.2]]))
print(clf.predict_proba([[0.2, 0.2]]))

# Regression
Xr = np.array([[0.], [1.], [2.], [3.]])
yr = np.array([0.0, 1.0, 1.5, 3.0])
reg = KNNRegressor(n_neighbors=2, weights="distance").fit(Xr, yr)
print(reg.predict([[1.5]]))
