# Community Detection

This module implements classic **graph community detection algorithms**.  
It is designed to be lightweight, readable, and suitable for educational use or small-scale experiments without external dependencies.

The code supports graphs represented as **dense adjacency or affinity matrices** and provides both spectral and iterative label-based approaches.

---

## Included Algorithms

### SpectralClustering
Clusters nodes by:
1. Constructing a graph Laplacian from an affinity matrix (or an RBF kernel over features),
2. Embedding nodes using the smallest eigenvectors of the Laplacian,
3. Applying a simple NumPy-based K-Means to the embedding.

Works with:
- Precomputed adjacency/affinity matrices
- Feature matrices (via RBF kernel)

---

### LabelPropagation
An iterative algorithm where each node repeatedly adopts the label that receives the **largest total edge weight** among its neighbors.

Key properties:
- No need to specify the number of communities in advance
- Random tie-breaking for stability
- Converges when labels stop changing or when `max_iter` is reached

---

## Quick Start

```python
import numpy as np
from rice_ml.unsupervised_learning.community_detection import (
    SpectralClustering,
    LabelPropagation,
)

# Example adjacency matrix
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 0],
], dtype=float)

# Spectral clustering
sc = SpectralClustering(n_clusters=2, random_state=0)
labels_sc = sc.fit_predict(A)

# Label propagation
lp = LabelPropagation(random_state=0)
labels_lp = lp.fit_predict(A)
````

---

## Design Notes

* Implemented **from scratch using NumPy only**
* Dense-matrix based (not optimized for very large graphs)
* Emphasizes clarity, correctness, and reproducibility
* Suitable for coursework, prototyping, and algorithm understanding

---

## Dependencies

* Python 3.9+
* NumPy

---