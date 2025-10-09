
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LinearRegression:
    def __init__(self):
        self.parameters = {}
        pass
    def forward(self, x):
        return self.parameters['w'] * x + self.parameters['b']
    def backward(self, x, y, y_pred):
        grad_w = -np.mean((y - y_pred) * x)
        grad_b = -np.mean(y - y_pred)
        return grad_w, grad_b
    def update(self, grad_w, grad_b, lr=0.01):
        self.parameters['w'] -= grad_w * lr
        self.parameters['b'] -= grad_b * lr
        pass
    def train(self, x, y, epochs=1000, batch_size=5, lr=0.01, figure=None, ax=None, line=None):
        self.parameters['w'] = 1.0
        self.parameters['b'] = 0.0
        def data_stream():
            for j in range(epochs):
                for i in range(0, len(x), batch_size):
                    x_batch = x[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                    y_batch_pred = self.forward(x_batch)
                    grad_w, grad_b = self.backward(x_batch, y_batch, y_batch_pred)
                    self.update(grad_w, grad_b, lr)
                loss = np.mean((y - self.forward(x)) ** 2)
                if j % 10 == 0:
                    print(f"w: {self.parameters['w']}, b: {self.parameters['b']}")
                    print(f"Epoch {j}, loss: {loss}")
                if j % 1 == 0:
                    yield float(self.parameters['w']), float(self.parameters['b'])
        def update_plot(frame):
            w, b = frame
            x1 = np.array([1, 10])
            y = w * x1 + b
            line.set_data(x1, y)
            return line,
        ani = FuncAnimation(figure, update_plot, frames=data_stream, interval=50, cache_frame_data=False, repeat=False)
        return ani
