import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 生成模拟数据
# 假设predictions是模型预测的结果（一维张量），labels是真实标签（一维张量）
# 为了演示目的，这里使用随机数据生成，实际应用中需要替换为实际数据
predictions = torch.randint(0, 10, (1000,))
labels = torch.randint(0, 10, (1000,))

# 计算混淆矩阵
conf_matrix = confusion_matrix(labels.numpy(), predictions.numpy())

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
