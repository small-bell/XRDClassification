import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xx = [i for i in range(5250)]
yy = [i for i in range(100)]
X, Y = np.meshgrid(xx, yy)
z = np.load('single.npy')

fig = plt.figure()
ax = plt.axes(projection='3d')

# 作图
ax.plot_surface(X, Y, z, cmap='rainbow')
plt.show()