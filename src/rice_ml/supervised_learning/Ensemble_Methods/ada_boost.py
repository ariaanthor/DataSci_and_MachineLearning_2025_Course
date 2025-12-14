from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class ADAClassifier:
    def __init__(self, max_depth, n_estimators, algorithm, learning_rate):
        self.classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.max_depth,), 
                             n_estimators = self.n_estimators,
                             algorithm = self.algorithm,
                             learning_rate = self.learning_rate)
        
def fit(self, X_train, y_train):
    self.classifier.fit(X_train, y_train)

def predict(self, X_test):
    return self.classifier.predict(X_test)