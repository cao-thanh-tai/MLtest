import numpy as np
import pickle 

class LinearRegression:
    def __init__(self, random_state = 42):
        self.W = None
        self.b = None
        self.random_state = random_state
        pass
    
    def _init_weights(self, n_feature):
        np.random.seed(self.random_state)
        self.W = np.random.rand(n_feature,1)*0.01
        self.b = 0.0
    
    def _forward(self, x) :
        return np.dot(x, self.W) + self.b
    
    def _backward(self, X, y, y_pred):
        error = y_pred - y
        n = X.shape[0]
        grad_w = (1/n) * np.dot(X.T, error)
        grad_b = (1/n)*np.sum(error)
        return grad_w, grad_b
    
    def _update(self, grad_w, grad_b, leaning_rate):
        self.W -= leaning_rate*grad_w
        self.b -= leaning_rate*grad_b
        
    
    def fit(self, X, y, batch_size=5, leaning_rate=0.01, epochs=1000, view_loss = False):
        
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1,1)
        n_samples, n_feature = X.shape
        
        
        if self.W is None:
            self._init_weights(n_feature)
        
            
        for j in range(epochs):
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                
                X_batch_size = X_shuffled[i:i + batch_size]
                y_batch_size = y_shuffled[i:i + batch_size]
                
                y_pred = self._forward(X_batch_size)
                grad_w, grad_b = self._backward(X_batch_size, y_batch_size, y_pred)
                self._update(grad_w, grad_b, leaning_rate)
                
            if view_loss:
                if j % 100 == 0 :
                    loss = np.mean((self._forward(X) - y)**2)
                    print(f"epoch : {j} = {loss }")
                    
    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if self.W is None :
            print("chua co mo hinh vui luong fit or load")
        return self._forward(X)
    
    def save_model(self, filepath):
        with open((f'src/weight/{filepath}'), 'wb') as f:
            pickle.dump({'W': self.W, 'b': self.b}, f)
            
    @staticmethod
    def load_model(filepath):
        model = LinearRegression()
        with open((f'src/weight/{filepath}'), 'rb') as f:
            params = pickle.load(f)
            model.W = params['W']
            model.b = params['b']
        return model