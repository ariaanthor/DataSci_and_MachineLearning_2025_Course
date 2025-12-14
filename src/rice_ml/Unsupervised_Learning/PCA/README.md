# Principal Component Analysis

This module implements **Principal Component Analysis (PCA)**. PCA is a linear dimensionality reduction technique that projects high-dimensional data onto a lower-dimensional subspace while preserving as much variance as possible.

The implementation closely follows standard PCA via **Singular Value Decomposition (SVD)** and is designed to be clear, dependency-free, and suitable for educational use or lightweight experimentation.

---

## Features

- Pure NumPy implementation (no external ML libraries)
- Supports:
  - Fixed number of components
  - Variance-based component selection (fraction of explained variance)
  - Full-rank PCA
- sklearn-like API: `fit`, `transform`, `fit_transform`, `inverse_transform`, `score`
- Access to explained variance and variance ratios
- Robust input validation and clear error handling

---

## Quick Start

```python
import numpy as np
from rice_ml.unsupervised_learning.pca import PCA

# Example dataset
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])

# Reduce to 2 principal components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Reduced shape:", X_reduced.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Reconstruct original data
X_reconstructed = pca.inverse_transform(X_reduced)
````

---

## Notes

* Data is **centered automatically** before decomposition.
* SVD is used instead of eigen-decomposition for numerical stability.
* When `n_components` is a float in `(0, 1)`, the smallest number of components
  explaining that fraction of total variance is selected.
* The `score` method returns the **negative reconstruction error** (higher is better),
  consistent with common PCA scoring conventions.

---

## Intended Use

This PCA implementation is well-suited for:

* Learning and teaching dimensionality reduction
* Prototyping pipelines without heavy dependencies
* Small to medium datasets where clarity and control matter
---
