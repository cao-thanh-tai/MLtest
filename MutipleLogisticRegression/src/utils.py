import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # ổn định
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_data(y):
    unique_y = np.unique(y)
    result = np.zeros((len(y), len(unique_y)))
    for i in range(len(y)):
        result[i][y[i]] = 1
    return result, unique_y
    
    
    
    