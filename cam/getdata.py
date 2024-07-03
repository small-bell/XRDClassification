import os

import numpy as np
from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, mat, label):
        self.mat = mat
        self.label = label

    def __getitem__(self, idx):
        xrd = self.mat[idx]
        label = self.label[idx]

        xrd = torch.tensor(xrd).unsqueeze(dim=0).to(torch.float32)
        label = torch.tensor(label).to(torch.long)

        return xrd, label

    def __len__(self):
        return len(self.label)


class CellDataset(Dataset):
    def __init__(self, root_dir='../data/spectra_data/'):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                file_path = os.path.join(cls_dir, file_name)
                samples.append((file_path, int(cls_name)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.loadtxt(file_path)
        data = torch.tensor(data).unsqueeze(dim=0).unsqueeze(dim=0).to(torch.float32)[0]
        label = torch.tensor(label).to(torch.long)
        return data, label
