
---

# K-Means Clustering

This notebook introduces **k-means clustering**, a foundational unsupervised learning algorithm that partitions data into a fixed number of clusters based on distance to cluster centroids.

## Contents

* Overview of unsupervised learning and clustering
* Intuition behind k-means (centroids, assignments, iterations)
* Applying k-means to a numeric dataset
* Visualizing cluster assignments and centroids
* Evaluating clustering quality (e.g., inertia / within-cluster sum of squares)
* Discussion of the role of `k` and common limitations

## Data

The notebook uses a **publicly available dataset with numeric features**, chosen to clearly illustrate how k-means groups data based on Euclidean distance. The data is lightly preprocessed and intended for visualization and conceptual understanding rather than domain-specific analysis.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn
* Matplotlib / Seaborn

## Usage

Run the notebook from top to bottom. All clustering, visualization, and evaluation steps are self-contained.

## Goal

The goal of this notebook is to provide a **clear introduction to centroid-based clustering**, while highlighting the assumptions and limitations of k-means, such as sensitivity to initialization and the need to choose the number of clusters in advance.

---
