
---
# CMOR 438 Data Science & Machine Learning 2025

### Name: Ariaan Ghatate
### NetID: ag201
---

This repository contains a **from-scratch machine learning library** along with a collection of **educational notebooks** and **tests** that demonstrate, evaluate, and validate core machine learning algorithms. The project is designed to emphasize **algorithmic understanding, correctness, and clarity** rather than reliance on black-box implementations.

---

## Repository Structure

```
.
├── src/
│   └── rice_ml/
│       ├── supervised_learning/
│       ├── unsupervised_learning/
│       └── processing/
├── notebooks/
│   ├── supervised_learning/
│   └── unsupervised_learning/
├── tests/unit
├── pyproject.toml
└── README.md
```

* **`src/rice_ml/`**
  Core Python package implementing machine learning algorithms from first principles, including both supervised and unsupervised methods.

* **`examples/`**
  Jupyter notebooks demonstrating the behavior, usage, and evaluation of implemented models on real and synthetic datasets.

* **`tests/unit`**
  Unit tests validating correctness and expected behavior of the implemented algorithms.

* **`pyproject.toml`**
  Project configuration and dependency management.

---

## Implemented Methods

The `rice_ml` package includes implementations of core machine learning algorithms, such as:

### Supervised Learning

* Linear regression
* Logistic regression
* k-Nearest Neighbors (kNN)
* Perceptron
* Regression trees
* Ensemble methods
* Neural networks

### Unsupervised Learning

* K-Means clustering
* DBSCAN
* Principal Component Analysis (PCA)
* Community detection

---

## Design Philosophy

This project prioritizes:

* **Algorithmic transparency** over black-box abstraction
* **Readable, well-structured code** suitable for learning and inspection
* **Separation of concerns** between implementation, experimentation, and testing
* **Reproducibility** through deterministic behavior and explicit configuration

---

## Installation

From the repository root:

```bash
pip install -e .
```

This installs the `rice_ml` package in editable mode.

---

## Running Tests

All tests are located in the `tests/` directory and can be executed using:

```bash
python -m pytest
```

---

## Purpose

This repository serves as:

* A **learning-focused machine learning library**
* A **companion codebase** for coursework or independent study
* A **reference implementation** of classical ML algorithms

---
