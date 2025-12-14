# K-Means Clustering

This module implements the K-Means clustering algorithm. It is designed to be lightweight, easy to read, and suitable for educational use or small-to-medium datasets where full-featured external libraries are unnecessary.

The implementation supports multiple centroid initialization strategies and repeated runs to improve clustering stability, closely mirroring the behavior of common machine learning libraries while remaining dependency-free.

---

## Features

- Pure NumPy implementation (no external ML libraries)
- Supports **k-means++** and **random** centroid initialization
- Multiple initializations (`n_init`) with best-inertia selection
- Early stopping based on centroid movement tolerance
- sklearn-like API: `fit`, `predict`, `fit_predict`, `transform`, `score`

---

## Quick Start

```python
import numpy as np
from rice_ml.unsupervised_learning.k_means_clustering import KMeans

# Sample data
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [9.0, 9.0],
    [10.0, 9.0]
])

# Fit K-Means
kmeans = KMeans(n_clusters=2, init="k-means++", random_state=42)
kmeans.fit(X)

# Cluster assignments
print(kmeans.labels_)

# Predict new points
print(kmeans.predict([[0.2, 0.1], [9.5, 9.2]]))

# Distance to cluster centers
print(kmeans.transform(X))
````

---

## Notes

* Distances are computed using squared Euclidean distance for efficiency.
* Empty clusters are automatically re-initialized to random data points.
* The `score` method returns **negative inertia**, matching scikit-learn conventions.

---

## Intended Use

This implementation is ideal for:

* Learning and teaching clustering algorithms
* Prototyping and experimentation
* Environments where minimizing dependencies is important

It is **not** optimized for very large datasets or high-dimensional data compared to highly optimized libraries like scikit-learn.

---