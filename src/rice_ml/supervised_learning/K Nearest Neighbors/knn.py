from util import distance

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for point in X_test:
            neighbors = self._k_nearest(point)
            votes = [label for _, label, _ in neighbors]
            pred = max(set(votes), key=votes.count)
            predictions.append(pred)
        return predictions

    def _k_nearest(self, point):
        neighbors = []
        for x_train, label in zip(self.X_train, self.y_train):
            d = distance(point, x_train)
            neighbors.append((x_train, label, d))
        neighbors.sort(key=lambda x: x[-1])
        return neighbors[:self.k]
