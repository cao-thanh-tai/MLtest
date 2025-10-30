import numpy as np
import pandas as pd

from src.LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as sklr

X_train = pd.read_csv('data/train.csv')
X_test = pd.read_csv('data/test.csv')

S_train=X_train.drop(['Survived',"Name","Age","Cabin","PassengerId","Embarked","Ticket"], axis= 1)
col = S_train.columns
S_train=S_train.to_numpy()
print(col)
y_train=X_train['Survived'].to_numpy()

S_test=X_test.drop(["Name","Age","Cabin","PassengerId","Embarked","Ticket"], axis= 1).to_numpy()

model = sklr()
model.fit(X=S_train,y=y_train)