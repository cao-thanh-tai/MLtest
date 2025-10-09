import Logistic_Regreession as lgr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_data():
    df = pd.read_excel('dataset_LGR.xlsx')
    x = df[['x']].values
    y = df[['y']].values
    return x, y
def load_data_2_chieu():
    df = pd.read_excel('dataset_LGR.xlsx')
    x = df[['x1','x2']].values
    y = df[['y']].values
    return x, y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
model = lgr.LogisticRegression()
X, Y = load_data_2_chieu()
print(X)
print(Y.shape)
W = model.train(X, Y, epochs=200, batch_size=5, lr=0.1)
# print(W)

fig, ax = plt.subplots()
line, = ax.plot([], [], color='red')

ax.scatter(X[:,0], X[:,1], c=Y.flatten(), cmap='bwr', alpha=0.5)
# ax.plot([0, 10], [0.5, 0.5], color='green')
print(len(W))
def update(frame):
    x = np.linspace(0, 10, 100)
    w = W[frame*10]
    y = sigmoid(w['w'] * x + w['b'])
    line.set_data(x, y)
    return line,

# ani = animation.FuncAnimation(fig, update, frames=int(len(W)/10), interval=50, cache_frame_data=False, repeat=False)
kq=model.predict(np.array([[1,0.5]]))
print(kq[0][0])
ax.scatter(1,0.5, c=["red" if kq[0][0]==1 else "blue"], alpha=1, label="diem moi")
ax.legend()
plt.show()
plt.close()