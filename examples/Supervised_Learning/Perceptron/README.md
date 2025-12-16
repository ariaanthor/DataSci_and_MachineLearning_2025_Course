
---

# Perceptron

This notebook demonstrates the **Perceptron algorithm** using a **from-scratch implementation provided by the `rice_ml` package**. The emphasis is on understanding the learning rule, model behavior, and limitations of linear classifiers rather than relying on black-box libraries.

## Contents

* Overview of the Perceptron algorithm and linear decision boundaries
* Description of the custom `rice_ml` Perceptron implementation
* Training the model using labeled data
* Visualizing or interpreting classification behavior
* Evaluating classification performance on test data
* Discussion of convergence and separability assumptions

## Data

The notebook uses a **publicly available binary classification dataset** with numeric features and class labels. The data is lightly preprocessed and split into training and testing sets to clearly illustrate how the Perceptron updates its weights based on misclassified examples.

## Requirements

* Python 3.x
* NumPy
* Pandas
* Matplotlib
* `rice_ml` (custom package included with the project)

## Usage

Ensure the `rice_ml` package is available on the Python path. Run the notebook sequentially from top to bottom to reproduce the training and evaluation results.

## Goal

The goal of this notebook is to provide a **clear, algorithm-level understanding** of the Perceptron, highlighting both its strengths and its limitations compared to more expressive models.

---
