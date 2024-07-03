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
        data = np.expand_dims(data, axis=0)
        return (data, label)

    def get_datas(self):
        for file_path, label, finename in tqdm(self.samples):
            data = np.loadtxt(file_path)
            data = np.expand_dims(data, axis=0)

            self.datas.append((data, label, finename))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label, filename = self.datas[idx]
        data = torch.tensor(data).to(torch.float32)
        label = torch.tensor(label).to(torch.long)
        return data, label, filename


if __name__ == '__main__':

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

    ############################################################
    for i in range(10):
        checkpoint = torch.load("./result/{}/model_results/checkpoints/ckpt_best.pth".format(i))

        net = SEResNet(num_classes=len(crystal_systems_result[i]))
        net = net.to(device)
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net.load_state_dict(checkpoint['net'], strict=True)

        net.eval()
        df = pd.DataFrame()
        for xrd, label, filename in tqdm(dl, desc='valid'):
            out, conf = net(xrd)
            pred_y = out.detach().argmax(dim=1)
            record = []
            record.append(filename[0]) # filename
            record.append(label.cpu().detach().numpy()[0]) # label
            record.append(crystal_systems_result[i][pred_y]) # 预测整体的label
            record.append(pred_y.cpu().detach().numpy()[0])  # 预测单个的label
            record.append(conf.detach().cpu().numpy()[0][0]) # conf
            record.append(out[0][pred_y[0]].cpu().item()) # 预测res概率
            record.append("TAG")

            for o in out.detach().cpu().numpy()[0]:
                record.append(o)

            df = df.append([record])

            # print("{}.conf={};pred_y={};rate={};real={}".format(i, conf.detach().cpu().numpy()[0][0],
            #                                                     pred_y.cpu().detach().numpy()[0], out[0][pred_y[0]],
            #                                                     crystal_systems_result[i][pred_y]))
        df.to_csv('res/{}-model.csv'.format(i), index=False, header=False)




