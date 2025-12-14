# DBSCAN Clustering

This module implements **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).  
DBSCAN groups points based on local point density and explicitly identifies **outliers** (noise), without requiring the number of clusters in advance.

The implementation is designed for clarity and correctness, mirroring the core logic of standard DBSCAN while avoiding external dependencies.

---

## Algorithm Overview

DBSCAN classifies points into three categories:
- **Core points**: have at least `min_samples` neighbors within distance `eps`
- **Border points**: reachable from a core point but not core themselves
- **Noise points**: not reachable from any core point (labeled `-1`)

Clusters are grown by iteratively expanding neighborhoods from core points.

---

## Features

- Pure **NumPy implementation**
- Supports **Euclidean** and **Manhattan** distance metrics
- Explicit noise labeling (`-1`)
- Compatible with 1D or multi-dimensional feature inputs
- scikit-learn–style API (`fit`, `fit_predict`)

---

## Quick Start

```python
import numpy as np
from rice_ml.unsupervised_learning.dbscan import DBSCAN

X = np.array([
    [1, 2],
    [2, 2],
    [2, 3],
    [8, 7],
    [8, 8],
    [25, 80],
], dtype=float)

db = DBSCAN(eps=3.0, min_samples=2)
labels = db.fit_predict(X)

print(labels)
# Example output: [ 0  0  0  1  1 -1 ]
````

---

## Parameters

* `eps`: Neighborhood radius
* `min_samples`: Minimum points required to form a dense region
* `metric`: Distance metric (`"euclidean"` or `"manhattan"`)

---

## Notes

* Uses a **dense pairwise distance matrix** → best suited for small to medium datasets
* Time complexity is roughly **O(n²)** due to distance computation
* Intended for education, prototyping, and algorithm understanding rather than large-scale production use

---

## Dependencies

* Python 3.9+
* NumPy
---