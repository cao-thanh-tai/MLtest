import numpy as np


class MultinomialLogisticRegression:
    def __init__(self, num_classes):
        self.parameters = {}
        self.num_classes = num_classes
        pass
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    def forward(self, x):
        return self.softmax(np.dot(x, self.parameters['W']) + self.parameters['b'])  #x.shape=(m,n), W.shape=(n,C), b.shape=(C,)
    def backward(self, x, y, y_pred):
        m = y.shape[0]
        grad_W = np.dot(x.T, (y_pred - y)) / m
        grad_b = np.mean(y_pred - y, axis=0)
        return grad_W, grad_b
    def update(self, grad_W, grad_b, lr=0.01):
        self.parameters['W'] -= grad_W * lr
        self.parameters['b'] -= grad_b * lr
        pass
    def train(self, x, y, epochs=1000, batch_size=5, lr=0.01):
        self.parameters['W'] = np.random.rand(x.shape[1], self.num_classes)*0.01
        self.parameters['b'] = np.zeros(self.num_classes)
        W = [self.parameters.copy()]
        for j in range(epochs):
            for i in range(0, len(x), batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_batch_pred = self.forward(x_batch)
                grad_W, grad_b = self.backward(x_batch, y_batch, y_batch_pred)
                self.update(grad_W, grad_b, lr)
            loss = -np.mean(np.sum(y * np.log(self.forward(x)), axis=1))
            if j % 10 == 0:
                print(f"W: {self.parameters['W']}, b: {self.parameters['b']}")
                print(f"Epoch {j}, loss: {loss}")
            if j % 1 == 0:
                W.append(self.parameters.copy())
        return W