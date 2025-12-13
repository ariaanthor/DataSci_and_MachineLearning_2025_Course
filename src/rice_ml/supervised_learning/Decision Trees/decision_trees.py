from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class DecisionTree:
    def __init__(self, max_depth, is_classifier = True):
    
        self.is_classifier = is_classifier

        if self.is_classifier:
            self.tree = DecisionTreeClassifier(max_depth=max_depth)
        else:
            self.tree = DecisionTreeRegressor(max_depth=max_depth)
        
    def fit(self, X_train, y_train):
        self.tree.fit(X_train, y_train)