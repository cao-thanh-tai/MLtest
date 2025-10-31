import numpy as np

class StandarkScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        pass
    
    def fit(self, X):
        self.mean_ = np.mean(X,axis=0)
        self.std_ = np.std(X, axis=0, ddof=1)
        self.std_[self.std_ == 0] = 1
    
    def transform(self, X):
        return (X - self.mean_)/self.std_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)