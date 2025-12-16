
---

# k-Nearest Neighbors (kNN)

This notebook demonstrates the **k-Nearest Neighbors (kNN)** algorithm using a **custom implementation from the `rice_ml` package**, rather than a library-provided model. The focus is on understanding the algorithmic mechanics behind kNN and validating correctness through empirical results.

## Contents

* Overview of the kNN algorithm and distance-based learning
* Description of the custom `rice_ml` kNN implementation
* Training and prediction using the custom model
* Effect of the parameter `k` on model behavior
* Evaluation of classification performance on held-out data
* Comparison to expected kNN behavior

## Data

The notebook uses a **publicly available classification dataset** split into training and test sets (`cs-training.csv` and `cs-test.csv`). The data consists of numeric features with a discrete class label and requires minimal preprocessing, making it well-suited for evaluating a distance-based classifier.

## Requirements

* Python 3.x
* NumPy
* Pandas
* Matplotlib
* `rice_ml` (custom package included with the project)

## Usage

Ensure the `rice_ml` package is available on the Python path. Run the notebook sequentially from top to bottom to reproduce training, prediction, and evaluation results.

## Goal

The goal of this notebook is to provide a **transparent, from-scratch demonstration** of kNN, emphasizing algorithmic understanding over black-box model usage.

---
