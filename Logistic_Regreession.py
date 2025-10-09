import numpy as np



class LogisticRegression:
    def __init__(self):
        self.parameters = {}
        pass
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def forward(self, x):
        return self.sigmoid(np.dot(x, self.parameters['w']) + self.parameters['b'])
    def backward(self, x, y, y_pred):
        grad_w = np.mean((y_pred - y) * x)
        grad_b = np.mean(y_pred - y)
        return grad_w, grad_b
    def update(self, grad_w, grad_b, lr=0.01):
        self.parameters['w'] -= grad_w * lr
        self.parameters['b'] -= grad_b * lr
        pass
    def train(self, x, y, epochs=1000, batch_size=5, lr=0.01):
        self.parameters['w'] = np.zeros((x.shape[1], 1))
        self.parameters['b'] = 0.0
        W = [self.parameters.copy()]
        for j in range(epochs):
            for i in range(0, len(x), batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_batch_pred = self.forward(x_batch)
                grad_w, grad_b = self.backward(x_batch, y_batch, y_batch_pred)
                self.update(grad_w, grad_b, lr)
            loss = -np.mean(y * np.log(self.forward(x)) + (1 - y) * np.log(1 - self.forward(x)))
            if j % 10 == 0:
                print(f"w: {self.parameters['w']}, b: {self.parameters['b']}")
                print(f"Epoch {j}, loss: {loss}")
            if j % 1 == 0:
                W.append(self.parameters.copy())
        return W

    def predict(self, x):
        y_pred = self.forward(x)
        return (y_pred >= 0.5).astype(int)