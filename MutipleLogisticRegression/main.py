import numpy as np
import pandas as pd


from src.MutipleLogisticRegression import MutipleLogisticRegression as mlgr


S = pd.read_csv("data/train.csv")

X = S.drop('y', axis=1).to_numpy()
y = S['y'].to_numpy()

model = mlgr()
model.fit(X, y, epochs=1000, lr = 0.01)
y_pred = model.predict(X)

print(y_pred)

mask = y_pred == y
print(mask)
print(np.sum(mask)/len(mask))