import numpy as np
import os

import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

from models.SEResNet import SEResNet
import warnings

# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore')


class CellMemoryDataset(Dataset):
    def __init__(self, root_dir='./test/'):
        self.root_dir = root_dir
        self.classes = [i for i in range(100)]
        self.types_num = len(self.classes)
        self.samples = self._make_dataset()
        self.datas = []
        self.get_datas()

    def _make_dataset(self):
        samples = []
        for cls_name in self.classes:
            try:
                cls_dir = os.path.join(self.root_dir, str(cls_name))
                for file_name in os.listdir(cls_dir):
                    file_path = os.path.join(cls_dir, file_name)
                    samples.append((file_path, int(cls_name), file_name))
            except:
                pass
        return samples

    def load_and_process_data(self, file_path, label):
        data = np.loadtxt(file_path)
        return (data, label)

    def get_datas(self):
        for file_path, label, finename in tqdm(self.samples):
            data = np.loadtxt(file_path)
            self.datas.append((data, label, finename))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label, filename = self.datas[idx]
        data = data
        label = label
        return data, label, filename


def cosine_similarity(vector1, vector2):
    """
    计算两个向量之间的余弦相似度。

    参数：
    vector1: 第一个向量，NumPy数组。
    vector2: 第二个向量，NumPy数组。

    返回值：
    两个向量之间的余弦相似度。
    """
    # 计算向量的点积
    dot_product = np.dot(vector1, vector2)

    # 计算向量的范数
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

    return cosine_similarity


if __name__ == '__main__':
    sum_matrix = np.load('npy/sum.npy')

    ds = CellMemoryDataset()
    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_works = 1
    batch_size = 1

    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_works,
                    pin_memory=True)
    mat_sum_df = pd.DataFrame()
    for xrd, label, filename in tqdm(dl, desc='valid'):
        current_sim = []
        current_sim.append(label.detach().cpu().numpy()[0])
        current_sim.append(filename[0])
        for item in sum_matrix:
            current_sim.append(cosine_similarity(item, xrd.reshape(5250,)))
        mat_sum_df = mat_sum_df.append([current_sim])

    mat_sum_df.to_csv("res/matsum.csv", index=False, header=False)

    single_matrix = np.load('npy/single.npy')

    mat_single_df = pd.DataFrame()
    for xrd, label, filename in tqdm(dl, desc='valid'):
        current_sim = []
        current_sim.append(label.detach().cpu().numpy()[0])
        current_sim.append(filename[0])
        for item in single_matrix:
            current_sim.append(cosine_similarity(item, xrd.reshape(5250, )))
        mat_single_df = mat_single_df.append([current_sim])

    mat_single_df.to_csv("res/matsingle.csv", index=False, header=False)

    print(mat_single_df.shape)