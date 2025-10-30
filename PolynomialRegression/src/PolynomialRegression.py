import numpy as np

class PolynomialRegression:
    def __init__(self, random_seed = 42, degree = 2):
        self.W = None
        self.degree = degree
        self.combos = None
        pass
    
    
    def polynomial_feature(self, X):
        n_sample, n_feature = X.shape
        if self.combos is None:
            self._set_combinations(n_feature)
        result = np.ones((n_sample,1))
        for combo in self.combos:
            new_col = np.prod(X**combo, axis=1).reshape(-1,1)
            result = np.hstack((result,new_col))
        return result
                
                
    def _set_combinations(self, n_feature):
        self.combos = []
        def backtrack(start, ar, curent_sum):
            if curent_sum > 0:
                self.combos.append(ar.copy())
            
            for i in range(start, n_feature):
                if curent_sum + 1 > self.degree:
                    continue
                
                ar[i] += 1
                backtrack(i, ar, curent_sum + 1)
                ar[i] -= 1
                
        backtrack(0, ar = [0]*n_feature, curent_sum=0)
        
        
        
    
    
        
    
    