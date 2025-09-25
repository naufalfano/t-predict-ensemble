import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict

class SuperLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learners, meta_learner, n_folds=5):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.n_folds = n_folds
    
    #Fit on out-of-fold predictions
    def fit(self, X, y):
        meta_ft = np.column_stack([
            cross_val_predict(learner, X, y, cv=self.n_folds, method='predict_proba')[:, 1]
            for learner in self.base_learners
        ])

        #Fit all base learners on full training set
        for learner in self.base_learners:
            learner.fit(X, y)
        
        #Fit meta learner
        self.meta_learner.fit(meta_ft, y)
        return self
    
    def predict(self, X):
        meta_ft = np.column_stack([
            learner.predict_proba(X)[:, 1]
            for learner in self.base_learners
        ])
        return self.meta_learner.predict(meta_ft)
    
    def predict_proba(self, X):
        meta_ft = np.column_stack([
            learner.predict_proba(X)[:, 1]
            for learner in self.base_learners
        ])
        return self.meta_learner.predict_proba(meta_ft)