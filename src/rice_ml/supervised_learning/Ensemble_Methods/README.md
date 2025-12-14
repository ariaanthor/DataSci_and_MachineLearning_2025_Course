# Ensemble Methods

This module implements a small set of **ensemble learning algorithms** using the package’s **from-scratch `DecisionTreeClassifier`** (no scikit-learn dependency). The goal is to provide readable, debuggable reference implementations of common ensemble ideas: **bagging**, **random forests**, and **boosting**.

---

## What’s Included

### `RandomForestClassifier`
An ensemble of decision trees trained on **bootstrap samples** (optional) with **random feature selection** at split time. Predictions are made by **averaging class probabilities** across trees and taking `argmax`.

### `BaggingClassifier`
A generic bagging-style classifier that trains multiple trees on:
- a random subset of samples (bootstrap or subsample)
- a random subset of features per estimator  
Final prediction is by **majority vote**.

### `AdaBoostClassifier`
A boosting classifier using **decision stumps** (depth-1 trees) as weak learners. It follows a **SAMME-style** multiclass boosting update and combines learners with a **weighted vote**.

---

## Quick Start

```python
import numpy as np
from rice_ml.supervised_learning.ensemble_methods import RandomForestClassifier

X = np.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]])
y = np.array([0, 0, 1, 1])

clf = RandomForestClassifier(n_estimators=5, random_state=0)
clf.fit(X, y)

print(clf.predict([[0., 0.], [1., 1.]]))
print(clf.predict_proba([[0., 0.], [1., 1.]]))
