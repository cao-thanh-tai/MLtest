import numpy as np


class K_NearestNeighbors:
    def __init__(self, k_neighbors = 5):
        self.k = k_neighbors
        self.X_train = None
        self.y_train= None
        self.categorical = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.categorical = self.is_categorical(y_train)
    
    def is_categorical(self, y):
        unique_values = np.unique(y)
        if (isinstance(y[0], (int, float))):
            return isinstance(y[0], int) and len(unique_values) < 10
        return True

    def predict(self, X):
        
        result = []
        
        for x in X:
            distance = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            k_indices = np.argpartition(distance,self.k)[:self.k]
            k_nearest_label = self.y_train[k_indices]
            
            if self.categorical:
                pred = np.bincount(k_nearest_label).argmax()
            else:
                pred = np.mean(k_nearest_label)
            result.append(int(pred))
        return result
            