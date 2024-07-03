import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 指定数据文件夹路径
from tqdm import tqdm

data_folder = "../data/spectra_data/"

# 获取文件夹列表
folder_list = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]

# 保存所有文件夹中的均值
all_means = []

# 遍历每个文件夹
for folder in tqdm(folder_list):
    folder_path = os.path.join(data_folder, folder)
    file_list = os.listdir(folder_path)
    # 保存每个文件夹中的数据
    folder_data = []
    # 遍历文件夹中的每个文件
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        # 使用np.loadtxt读取数据文件
        data = np.loadtxt(file_path)
        folder_data.append(data)
    # 计算每个文件夹中数据的均值
    folder_data_mean = np.mean(np.array(folder_data), axis=0)
    if len(folder_data) > 0:
        all_means.append(folder_data_mean)

# 转换成numpy数组
all_means = np.array(all_means)

np.save('single.npy', all_means)
