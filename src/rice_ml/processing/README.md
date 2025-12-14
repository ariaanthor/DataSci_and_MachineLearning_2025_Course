# Processing Utilities

This directory contains **data processing utilities** used throughout the `rice_ml` library. The modules are designed to be lightweight, dependency-free, and consistent with scikit-learn–style APIs while remaining fully transparent and easy to inspect.

## Contents

### Preprocessing

Utilities for preparing datasets before model training, including scaling, normalization, and dataset splitting .

**Key features:**

* Feature scaling and normalization:

  * Z-score standardization
  * Min–max scaling
  * Max-absolute scaling
  * Row-wise L2 normalization
* Dataset splitting:

  * Train/test split
  * Train/validation/test split
  * Optional shuffling and stratification
* Robust input validation and reproducible randomness

### Postprocessing & Evaluation

Utilities for model output aggregation and performance evaluation after training .

**Key features:**

* Classification metrics:

  * Accuracy, precision, recall, F1
  * Confusion matrix
  * ROC AUC (binary)
  * Log loss (binary and multiclass)
* Regression metrics:

  * MSE, RMSE, MAE
  * R² score
* Simple decision aggregation helpers (e.g., majority voting, distance-weighted averaging)

## Design Philosophy

* **No external ML dependencies
* **Explicit validation** to surface shape and type errors early
* **Test-friendly**: functions include doctest-style examples
* **Educational clarity**: readable implementations intended for learning and extension

These utilities are meant to serve as building blocks for supervised learning pipelines while remaining transparent and easy to modify for research or instructional use.
