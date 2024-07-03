import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建数据
x = [i for i in range(5250)]
y = [i for i in range(100)]
x, y = np.meshgrid(x, y)  # 创建网格
z = np.load('single.npy')

# 创建三维图形窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维曲线图
ax.plot_surface(x, y, z, cmap='viridis')

# 设置图形的标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()
