import numpy as np
import pandas as pd

from src.LogisticRegression import LogisticRegression as lgr
from sklearn.linear_model import LogisticRegression as sklgr


S = pd.read_csv('data/train.csv')

X = S.drop('y', axis=1).to_numpy()
y = S['y'].to_numpy()

model = lgr()

model.fit(X,y,epochs=10000)

y_pred = model.predict(X)

mask = y_pred == y
for i in range(len(mask)):
    if mask[i] == 0:
        print(i)
        break
print(np.sum(mask)/len(mask))

model1 = sklgr()
model1.fit(X,y)
y_pred_sk = model1.predict(X)

mask1 = y_pred_sk == y

for i in range(len(mask1)):
    if mask1[i] == 0:
        print(i)
        break
print(np.sum(mask1)/len(mask1))