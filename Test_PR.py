import Polynomial_Regression as pr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def load_data():
    df = pd.read_excel("dataset_PR.xlsx")
    x=df["x"].to_numpy()
    y=df["y"].to_numpy()
    return x, y


model=pr.PolynomialRegression(degree=3)

X, Y = load_data()

W = model.train(X, Y, epochs=100, batch_size=10, lr=0.01)

print(W)
print(model.parameters)
print(W[0])
print(len(W))
fig, ax = plt.subplots()
line, = ax.plot([], [], color='red')
ax.set_xlim(-4, 2)
ax.set_ylim(-4, 10)
def update(frame):
    x=np.linspace(-4, 2, 100)
    w=W[frame*10]
    y=0
    for i in range(len(w)):
        y += w[i] * x**i
    line.set_data(x , y)
    return line,
ani = animation.FuncAnimation(fig, update, frames=int(len(W)/10), interval=50, cache_frame_data=False, repeat=False)

ax.scatter(X, Y)
plt.show()
plt.close()
