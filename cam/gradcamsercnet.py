from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from SEResNet import SEResNet
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def reshape_transform(tensor):
    image_with_single_row = tensor[:, None, :, :]
    # Lets make the time series into an image with 16 rows for easier visualization on screen later
    target_size = 16, tensor.size(1)
    return torch.nn.functional.interpolate(image_with_single_row, target_size, mode='bilinear')


# idx = 9
# xrd = np.loadtxt("99/Gd3Ga5O12-mp-2921.cif0.x")
# xrd = np.loadtxt("99/Gd3Fe5O12-mp-557370.cif0.x")

idx = 0
# xrd = np.loadtxt("1/Ca7MgMn8O20-mp-1076897.cif0.x")
# xrd = np.loadtxt("0/Li4Mn3OF8-mp-764336.cif0.x")
# xrd = np.loadtxt("0/Li5Co4(Si3O10)2-mp-849656.cif0.x")
xrd = np.loadtxt("0/res.x")
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

plot_xrd = xrd
xrd = torch.tensor(xrd).unsqueeze(dim=0).unsqueeze(dim=0).to(torch.float32)

device_ids = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(r"../result/{}/model_results/checkpoints/ckpt_best.pth".format(idx))

net = SEResNet(num_classes=len(crystal_systems_result[idx]))
target_layers = [net.layer4, net.avgpool, net.maxpool, net.layer4[-1]]
net = net.to(device)
net = torch.nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(checkpoint['net'], strict=True)
net.eval()
out, conf = net(xrd)
pred_y = out.detach().argmax(dim=1)

print(pred_y)

with GradCAM(model=net, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
    grayscale_cam = cam(input_tensor=xrd)
    grayscale_cam = grayscale_cam[0, :]
    plt.figure()
    # plt.plot(xrd.squeeze().numpy())
    np.savetxt("target/res.txt", grayscale_cam.T.squeeze())

    plt.ylabel('XRD', color='b')
    plt.plot(grayscale_cam.T.squeeze(), color='red')

    plt.twinx()
    plt.ylabel('Grad', color='r')
    plt.plot(plot_xrd)
    plt.show()
