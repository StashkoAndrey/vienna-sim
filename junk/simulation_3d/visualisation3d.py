import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

def plot_scatter(grid, t):
    ax.cla()
    ax.set_title(f"Bacteria (3D Scatter) at t={t}")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    xs, ys, zs = [], [], []
    for z in range(10):
        for y in range(10):
            for x in range(10):
                if grid[z][y][x] is not None:
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

    ax.scatter(xs, ys, zs, c='green', marker='o')
    plt.pause(0.1)