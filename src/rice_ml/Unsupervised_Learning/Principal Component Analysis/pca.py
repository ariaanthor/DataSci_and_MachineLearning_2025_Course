from sklearn.decomposition import PCA
from sklearn import preprocessing

class PCA:
    def __init__(self, X):
        self.scaled_X = preprocessing.scale(X)
        self.pca = PCA()

    def fit(self):
        self.pca.fit(self.scaled_X)