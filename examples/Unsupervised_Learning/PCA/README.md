
---

# Principal Component Analysis (PCA)

This notebook introduces **Principal Component Analysis (PCA)**, a widely used unsupervised technique for **dimensionality reduction** and data visualization.

## Contents

* Motivation for dimensionality reduction
* Intuition behind PCA (variance maximization and orthogonal components)
* Applying PCA to high-dimensional numeric data
* Visualizing data in lower-dimensional subspaces
* Interpreting explained variance and principal components
* Discussion of trade-offs between dimensionality reduction and information loss

## Data

The notebook uses a **publicly available dataset with numeric features**, suitable for demonstrating dimensionality reduction and visualization. The data is standardized prior to applying PCA to ensure features contribute appropriately to the learned components.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn
* Matplotlib / Seaborn

## Usage

Run the notebook sequentially from top to bottom. All preprocessing, PCA computation, and visualization steps are self-contained.

## Goal

The goal of this notebook is to provide an **intuitive understanding of PCA**, showing how high-dimensional data can be compressed into a smaller number of informative components while preserving as much variance as possible.

---
