import os

import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

# crystal_systems = {
#     'triclinic': [1, 2],  # size:2
#     'monoclinic': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # size:13
#     'orthorhombic1': [19, 20, 25, 26, 29, 31, 33, 36, 38, 40],  # size:10
#     'orthorhombic2': [43, 44, 46, 47, 51, 52, 55, 57, 58, 59],  # size:10
#     'orthorhombic3': [60, 61, 62, 63, 64, 65, 69, 70, 71, 72, 74],  # size:11
#     'tetragonal1': [82, 87, 88, 92, 99, 107, 115, 119, 121, 122],  # size:10
#     'tetragonal2': [123, 127, 128, 129, 136, 137, 139, 140, 141, 142],  # size:10
#     'trigonal': [143, 146, 147, 148, 150, 155, 156, 160, 161, 164, 166, 167],  # size:12
#     'hexagonal': [173, 174, 176, 186, 187, 189, 191, 193, 194],  # size:9
#     'cubic': [198, 204, 205, 215, 216, 217, 220, 221, 223, 225, 227, 229, 230]  # size:13
# }

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

crystal_systems_result = []

for value in crystal_systems.values():
    crystal_systems_result.append(value)


class CellMemoryDataset(Dataset):
    def __init__(self, root_dir='../data/spectra_data/'):
        self.root_dir = root_dir
        self.classes = [i for i in range(10)]
        self.types_num = len(self.classes)
        self.map = {}
        for idx, (k, v) in enumerate(crystal_systems.items()):
            for i in v:
                self.map[i] = idx
        self.samples = self._make_dataset()
        self.datas = []
        self.get_datas()

    def _make_dataset(self):
        samples = []
        for cls_name in range(100):
            cls_dir = os.path.join(self.root_dir, str(cls_name))
            for file_name in os.listdir(cls_dir):
                file_path = os.path.join(cls_dir, file_name)
                samples.append((file_path, self.map[int(cls_name)]))
        return samples

    def load_and_process_data(self, file_path, label):
        data = np.loadtxt(file_path)
        data = np.expand_dims(data, axis=0)
        return (data, label)

    def get_datas(self):
        for file_path, label in tqdm(self.samples):
            data = np.loadtxt(file_path)
            data = np.expand_dims(data, axis=0)
            self.datas.append((data, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.datas[idx]
        data = torch.tensor(data).to(torch.float32)
        label = torch.tensor(self.map[label]).to(torch.long)
        return data, label


class CellDiskDataset(Dataset):
    def __init__(self, root_dir='../data/spectra_data/'):
        self.root_dir = root_dir
        self.classes = [i for i in range(10)]
        self.types_num = len(self.classes)
        self.map = {}
        for idx, (k, v) in enumerate(crystal_systems.items()):
            for i in v:
                self.map[i] = idx
        self.samples = self._make_dataset()

        print(self.map)

    def _make_dataset(self):
        samples = []
        for cls_name in range(100):
            try:
                cls_dir = os.path.join(self.root_dir, str(cls_name))
                for file_name in os.listdir(cls_dir):
                    file_path = os.path.join(cls_dir, file_name)
                    samples.append((file_path, self.map[int(cls_name)]))
            except:
                pass
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.loadtxt(file_path)
        data = torch.tensor(data).unsqueeze(dim=0).to(torch.float32)
        label = torch.tensor(label).to(torch.long)
        return data, label

