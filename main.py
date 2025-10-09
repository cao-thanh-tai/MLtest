import LinearRegression as lr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

def generate_data(w=2, b=3, n=50):
    with open("data.txt", "w") as f:
        for i in range(n):
            f.write(f"{i} ")
        f.write("\n")
        for i in range(n):
            f.write(f"{i*w + b} ")
def load_data():
    df = pd.read_excel("datatest.xlsx")
    x=df["x"].to_numpy()
    y=df["y"].to_numpy()
    return x, y
x1, y = load_data()

X=np.array(x1)
Y=np.array(y)

X=X.reshape(-1, 1)
Y=Y.reshape(-1, 1)

model = lr.LinearRegression()
# W, B = model.train(X, Y, epochs=1000)



fig, ax = plt.subplots()

line, = ax.plot([], [], color='red')


ani = model.train(X, Y, epochs=100, batch_size=5, lr=0.01, figure=fig, ax=ax, line=line)

ax.scatter(x1, y)

# ax.plot([1, 10], [model.w*1 + model.b, model.w*10 + model.b], color='red')


# def update(frame):
#     w=W[frame]
#     b=B[frame]
#     y=w*x + b
#     line.set_data(x, y)
#     return line,

# ani = FuncAnimation(fig, update, frames=len(W), interval=50)

plt.show()
plt.close()
