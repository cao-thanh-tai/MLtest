import numpy as np


a = np.array([[1, 2, 3],[1, 2, 3]])
b = np.array([[1], [2]])
print(a.shape)
print(b.shape)

print(np.mean(a, axis=1) - b)