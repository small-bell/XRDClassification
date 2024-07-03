import numpy as np
import matplotlib.pyplot as plt

# 示例数据
matrix = np.load('single.npy')  # (11, 5250) 形状的矩阵
np.savetxt('single.csv', matrix, delimiter=',')
# 创建图像和子图
plt.figure(figsize=(10, 6))
for i in range(matrix.shape[0]):  # 遍历矩阵的每一行
    plt.plot(matrix[i], label=f'Types {i+1}')  # 绘制每一行
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot Types')
plt.legend()
plt.show()
