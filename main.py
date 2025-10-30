import numpy as np
import pandas as pd




from LinearRegresion.src.LinearRegression import LinearRegression as linear
from PolynomialRegression.src.PolynomialRegression import PolynomialRegression as poly



S = pd.read_csv('PolynomialRegression/data/train3.csv')

X = S.drop('y',axis=1).to_numpy()

y = S['y'].to_numpy().reshape(-1,1)
pl = poly(degree=2)
lr = linear()
X_pol = pl.polynomial_feature(X)

lr.fit(X=X_pol, y=y)

y_pred = lr.predict(X_pol)

loss_mea = np.mean(abs(y.reshape(-1,1) - y_pred))
print(loss_mea)
print(lr.W)
