from processing import *
from unsupervised_learning import *
from supervised_learning import *

__version__ = "0.1.0"

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
    
    # KNN
    'KNNClassifier',
    'KNNRegressor',
    
    # Trees
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    
    # Linear Models
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'LogisticRegression',
    
    # Neural Networks
    'Perceptron',
    'MLPClassifier',
    'MLPRegressor',
    'RNNClassifier',
    'GRUClassifier',
    
    # Ensemble Methods
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'BaggingClassifier',
    
    # Clustering
    'KMeans',
    'DBSCAN',
    'SpectralClustering',
    'LabelPropagation',
    
    # Dimensionality Reduction
    'PCA',
]