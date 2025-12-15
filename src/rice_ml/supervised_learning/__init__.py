from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier
from rice_ml.supervised_learning.regression_trees import DecisionTreeRegressor
from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor
from rice_ml.supervised_learning.linear_regression import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
)
from rice_ml.supervised_learning.logistic_regression import LogisticRegression
from rice_ml.supervised_learning.perceptron import Perceptron
from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier, MLPRegressor
from rice_ml.supervised_learning.ensemble_methods import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from rice_ml.supervised_learning.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
)

__all__ = [
    # Decision Trees
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    # KNN
    'KNNClassifier',
    'KNNRegressor',
    # Linear Models
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'LogisticRegression',
    # Neural Networks
    'Perceptron',
    'MLPClassifier',
    'MLPRegressor',
    # Ensemble Methods
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'BaggingClassifier',
    # Distance Metrics
    'euclidean_distance',
    'manhattan_distance',
]