import numpy as np



from .utils import softmax, one_hot_data
 

class MutipleLogisticRegression:
    def __init__(self, random_seed = 42):
        self.W = None  
        self.b = None
        self.random_seed = random_seed
        self.classes = None
        pass
    
    def _init_parameters(self, n_feature, n_classes):
        np.random.seed(self.random_seed)
        self.W = np.random.randn(n_feature, n_classes)
        self.b = np.zeros(1, n_classes)
    
    def _forward(self, X):
        return softmax(X @ self.W + self.b)
    
    def _backward(self, X, y, y_pred):
        error = y_pred - y
        n = X.shape[0]
        
        grad_w = (1/n)*np.dot(X.T, error)
        grad_b = (1/n)*np.sum(error, axis=0)
        
        return grad_w, grad_b
    
    def _update(self, grad_w, grad_b, lr=0.01):
        self.W -= lr*grad_w
        self.b -= lr*grad_b
    
    def _cross_entropy_loss(self, y_pred, y_true):
        n = y_pred.shape[0]
        log_prob = -np.log(y_pred[range(n), y_true] + 1e-15)
        return np.sum(log_prob) / n
    
    def fit(self, X, y, batch_size=5, lr=0.01, epochs=1000, view_loss = False):
        X = np.array(X, dtype=np.float64)
        y = y.reshape(-1,1)
        n_samples, n_feature = X.shape
        y_one_hot, self.classes = one_hot_data(y)
        n_classes = y_one_hot.shape[1]
        
        
        if self.W is None:
            self._init_parameters(n_feature,n_classes)
        
            
        for j in range(epochs):
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]
            
            for i in range(0, n_samples, batch_size):
                
                X_batch_size = X_shuffled[i:i + batch_size]
                y_batch_size = y_shuffled[i:i + batch_size]
                
                y_pred = self._forward(X_batch_size)
                grad_w, grad_b = self._backward(X_batch_size, y_batch_size, y_pred)
                self._update(grad_w, grad_b, lr)
                
            if view_loss:
                if j % 100 == 0 :
                    y_pred_full = self._forward(X)
                    loss = self._cross_entropy_loss(y_pred_full, np.argmax(y_one_hot, axis=1))
                    print(f"epoch : {j} = {loss }")
                    
    def predict(self, X):
        y_pred = self._forward(X)
        predicted_indices = np.argmax(y_pred,axis=1)
        predicted_labels = [int(self.classes[i]) for i in predicted_indices]
        return predicted_labels
        
        
        