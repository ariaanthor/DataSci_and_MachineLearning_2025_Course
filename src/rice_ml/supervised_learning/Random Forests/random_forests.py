from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class RandomForest:
    def __init__(self, 
                 max_depth, 
                 n_estimators, 
                 bootstrap = True, 
                 n_jobs = -1, 
                 is_classifier = True):
        
        self.is_classifier = is_classifier

        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs

        if self.is_classifier:
            self.forest = RandomForestClassifier(max_depth=max_depth, 
                                                 n_estimators=n_estimators, 
                                                 bootstrap=bootstrap, 
                                                 n_jobs=n_jobs
                                                 )
        else:
            self.forest = RandomForestRegressor(max_depth=max_depth, 
                                                 n_estimators=n_estimators, 
                                                 bootstrap=bootstrap, 
                                                 n_jobs=n_jobs
                                                 )
            
    def train(self, X_train, y_train, X_test):
        if self.is_classifier:
            bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=self.max_depth,),
                                n_estimators = self.n_estimators,
                                bootstrap = self.bootstrap,
                                n_jobs = self.n_jobs)
        else:
            bag_clf = BaggingClassifier(DecisionTreeRegressor(max_depth=self.max_depth,),
                                n_estimators = self.n_estimators,
                                bootstrap = self.bootstrap,
                                n_jobs = self.n_jobs)
        bag_clf.fit(X_train, y_train)
        self.forest.fit(X_train, y_train)
        forest_y_pred = bag_clf.predict(X_test)
        return forest_y_pred
