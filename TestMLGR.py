import MultinomialLogisticRegressoin as mlgr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

def load_data():
    df = pd.read_excel("dataset_MLGR.xlsx")
    x=df[["x"]].values
    y=df["y"].to_numpy()

    return x, y

X, Y1 = load_data()
Y = np.zeros((Y1.size, 4))
for i in range(Y1.size):
    Y[i, Y1[i]-1] = 1
print(X)
print(Y)
print(Y1)
print(X.shape)
print(Y.shape)

model = mlgr.MultinomialLogisticRegression(num_classes=4)
W = model.train(X, Y, epochs=200, batch_size=5, lr=0.1)
print(len(W))
x=2.5
kq=np.argmax(model.forward(np.array([[x]])))+1
print(kq)

fig, ax = plt.subplots()
plt.scatter(X,Y1)

plt.scatter(x,kq,c='red')
plt.show()
plt.close()
