# Distance metrics
from rice_ml.supervised_learning.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
)

# Supervised learning
from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor
from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier
from rice_ml.supervised_learning.regression_trees import DecisionTreeRegressor
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

# Unsupervised learning
from rice_ml.unsupervised_learning.k_means_clustering import KMeans
from rice_ml.unsupervised_learning.dbscan import DBSCAN
from rice_ml.unsupervised_learning.pca import PCA
from rice_ml.unsupervised_learning.community_detection import (
    SpectralClustering,
    LabelPropagation,
)

__all__ = [
    # Version
    '__version__',
    
    # Preprocessing
    'standardize',
    'minmax_scale',
    'maxabs_scale',
    'l2_normalize_rows',
    'train_test_split',
    'train_val_test_split',
    
    # Post-processing (metrics)
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'roc_auc_score',
    'log_loss',
    'mse',
    'rmse',
    'mae',
    'r2_score',
    
    # Distance metrics
    'euclidean_distance',
    'manhattan_distance',
    
    # Supervised Learning - KNN
    'KNNClassifier',
    'KNNRegressor',
    
    # Supervised Learning - Trees
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    
    # Supervised Learning - Linear Models
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'LogisticRegression',
    
    # Supervised Learning - Neural Networks
    'Perceptron',
    'MLPClassifier',
    'MLPRegressor',
    
    # Supervised Learning - Ensemble
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'BaggingClassifier',
    
    # Unsupervised Learning - Clustering
    'KMeans',
    'DBSCAN',
    'SpectralClustering',
    'LabelPropagation',
    
    # Unsupervised Learning - Dimensionality Reduction
    'PCA',
]