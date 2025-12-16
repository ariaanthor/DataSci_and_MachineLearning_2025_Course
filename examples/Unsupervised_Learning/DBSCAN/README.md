
---

# DBSCAN Clustering

This notebook introduces **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**, an unsupervised learning algorithm designed to identify clusters of arbitrary shape while explicitly handling noise and outliers.

## Contents

* Overview of clustering and density-based methods
* Intuition behind DBSCAN (ε-neighborhoods, core points, border points, noise)
* Applying DBSCAN to a dataset with no ground-truth labels
* Visualizing discovered clusters and identified noise points
* Discussion of key hyperparameters (`eps`, `min_samples`) and their effects
* Comparison to centroid-based clustering methods

## Data

The notebook uses a **publicly available dataset with numeric features**, suitable for unsupervised clustering. The data is minimally preprocessed and primarily used to illustrate how DBSCAN discovers clusters based on point density rather than distance to a centroid, making it effective for irregularly shaped clusters and noisy data.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn
* Matplotlib / Seaborn

## Usage

Run the notebook sequentially from top to bottom. All clustering, visualization, and analysis steps are self-contained and require no external configuration.

## Goal

The goal of this notebook is to provide an **intuitive, hands-on understanding of density-based clustering**, emphasizing when DBSCAN is preferable to traditional clustering algorithms like k-means.

---
