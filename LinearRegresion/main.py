import numpy as np
import pandas as pd


from src.LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as sklr

model = LinearRegression()

S_train = pd.read_csv('data/train1.csv')
X_train = S_train.drop('y', axis=1).to_numpy()
y_train = S_train['y'].to_numpy()

# model.fit(X_train, y_train, view_loss=True)
# print(model.W)
# print(model.b)
# model.save_model('test1')

modelsk = sklr()
modelsk.fit(X=X_train,y=y_train)
print(modelsk.predict(X_train))