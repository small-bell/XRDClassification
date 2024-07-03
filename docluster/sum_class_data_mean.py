import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 指定数据文件夹路径
from tqdm import tqdm

data_folder = "../data/spectra_data/"

crystal_systems = {'triclinic': [0, 1],
                   'monoclinic': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                   'orthorhombic1': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                   'orthorhombic2': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                   'orthorhombic3': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45],
                   'tetragonal1': [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
                   'tetragonal2': [56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
                   'trigonal': [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77],
                   'hexagonal': [78, 79, 80, 81, 82, 83, 84, 85, 86],
                   'cubic': [87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]}

all_means = []

for v in tqdm(crystal_systems.values()):
    folder_data = []
    for i in v:
        dirs = os.listdir(os.path.join(data_folder, str(i)))
        for dir in dirs:
            data = os.path.join(os.path.join(data_folder, str(i)), dir)
            data = np.loadtxt(data)
            folder_data.append(data)

    folder_data_mean = np.mean(np.array(folder_data), axis=0)
    all_means.append(folder_data_mean)

all_means = np.array(all_means)

np.save('sum.npy', all_means)
