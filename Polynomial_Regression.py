import numpy as np


class PolynomialRegression:
    def __init__(self, degree=2):
        self.parameters = {}
        self.degree = degree
        pass
    #w=[0,0,0] for degree=2
    #y=w0 + w1*x + w2*x^2
    def forward(self, x):
        y_pred = np.zeros_like(x)
        for i in range(self.degree + 1):
            y_pred += self.parameters['w'][i] * (x ** i)
        return y_pred

    def backward(self, x, y, y_pred):
        loss = y - y_pred
        grad_w = np.zeros(self.degree + 1)
        for i in range(self.degree + 1):
            grad_w[i] = -np.mean(loss * (x ** i))
        return grad_w
    
    def update(self, grad_w, lr=0.01):
        self.parameters['w'] -= grad_w * lr
        pass

    def train(self, x, y, epochs=1000, batch_size=5, lr=0.01):
        self.parameters['w'] = np.zeros(self.degree + 1)
        W = [self.parameters['w']]
        for j in range(epochs):
            for i in range(0, len((x)), batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_batch_pred = self.forward(x_batch)
                grad_w = self.backward(x_batch, y_batch, y_batch_pred)
                self.update(grad_w, lr)
                W.append(self.parameters['w'].copy())
            if j % 100 == 0:
                loss = np.mean((y - self.forward(x)) ** 2)
                print(f"Epoch {j}, loss: {loss}")
        return np.array(W)