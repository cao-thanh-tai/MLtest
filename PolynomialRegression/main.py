import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.PolynomialRegression import PolynomialRegression
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures as pl
from sklearn.pipeline import make_pipeline 


S_train = pd.read_csv("data/test.csv")
# print(S_train)

X_train = S_train.drop('y', axis=1).to_numpy()
y_train = S_train['y'].to_numpy()


model = PolynomialRegression(degree=3)
print(model.polynomial_feature(X_train))




# model.fit(X=X_train, y=y_train, degree=3,view_loss=True, lr=0.0001, epochs=1000)
# print(model.W)

# polyreg = make_pipeline(pl(degree=3),lr())

# polyreg.fit(X=X_train,y=y_train)

# # 4. Dự đoán trên lưới điểm mịn để vẽ đường cong
# X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
# y_plot = polyreg.predict(X_plot)

# # 5. Vẽ kết quả
# plt.figure(figsize=(8, 6))
# plt.scatter(X_train, y_train, color='blue', label='Dữ liệu thực tế', alpha=0.6)
# plt.plot(X_plot, y_plot, color='red', label=f'Polynomial Regression (bậc {3})')
# plt.title('Polynomial Regression với sklearn')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()





