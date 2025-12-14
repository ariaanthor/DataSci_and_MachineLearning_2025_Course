import numpy as np
from util import *

class KMC:
    def __init__(self, X, k):
        self.X = X
        self.k = k

    def cluster(self, max_iter):
        centers = []
        for j in range(self.k):
            i = np.random.randint(0, 100)
            point = (self.X[i, 0], self.X[i, 1], j)
            centers.append(point)
        
        for _ in range(max_iter):
            centers = update_centers(self.X, centers)

        return centers