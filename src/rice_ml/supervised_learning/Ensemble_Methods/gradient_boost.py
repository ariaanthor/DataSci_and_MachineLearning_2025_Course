from sklearn.ensemble import GradientBoostingRegressor

class GradientRegressor:
    def __init__(self, max_depth, n_estimators, learning_rate):
        self.regressor = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)

    def fit(self, X_train, y_train):
        self.regressor.fit(X_train, y_train)

    def predict(self, X_new):
        return self.regressor.predict(X_new)