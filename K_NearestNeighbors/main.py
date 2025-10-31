import numpy as np
import pandas as pd

from src.K_NearestNeighbors import K_NearestNeighbors as KNN
from sklearn.linear_model import LogisticRegression as sklgr


S = pd.read_csv('data/train.csv')

X = S.drop('y', axis=1).to_numpy()
y = S['y'].to_numpy()

model = KNN(k_neighbors=5)

model.fit(X_train=X, y_train=y)
y_pred = model.predict(X)

mask = y_pred == y
print(mask)
print(np.sum(mask)/len(mask))

