


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.scale_ = None
        pass
    
    def fit(self, X):
        self.min_ = X.min(axis = 0)
        self.scale_ = X.max(axis = 0) - self.min_
        self.scale_[self.scale_ == 0] = 1
        
    def transform(self, X):
        return (X - self.min_)/self.scale_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)