import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# dữ liệu x
x = np.linspace(0, 10, 100)

# tạo figure
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)  # y_max = 10*10

line, = ax.plot([], [], 'b-', lw=2)  # đường thẳng ban đầu

# list hệ số a
a_list = np.arange(1, 11)  # từ 1 đến 10
b_list = np.arange(1, 11)  # từ 1 đến 10
x = np.array([1, 10])

# hàm update
def update(frame):
    a = a_list[frame]
    b = b_list[frame]
    y = a * x + b
    line.set_data(x, y)
    return line,

# tạo animation
ani = FuncAnimation(fig, update, frames=len(a_list), interval=500, blit=True)
# fig.plot()

plt.show()
plt.close()