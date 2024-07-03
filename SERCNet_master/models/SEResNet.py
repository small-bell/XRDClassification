import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetPlus3(nn.Module):

    def __init__(self, block, layers, num_classes=18):
        self.inplanes = 64
        super(ResNetPlus3, self).__init__()

        self.expland_channel = nn.Conv1d(1, 12, kernel_size=15, stride=2, padding=7,
                                         groups=1, bias=False)

        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(12, 64, kernel_size=31, stride=2, padding=15,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(12, 64, kernel_size=45, stride=2, padding=22,
                               bias=False)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(12, 64, kernel_size=9, stride=2, padding=4,
                               bias=False)
        self.bn4 = nn.BatchNorm1d(64)

        self.conv11 = nn.Conv1d(64, 16, kernel_size=1, bias=False)
        self.se = SELayer(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(512 * block.expansion * 2, 128),
            Hswish(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.dense2 = nn.Linear(num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.expland_channel(x)
        x1 = self.conv1(x)
        # print(x1.shape)
        x1 = self.bn1(x1)
        # print(x1.shape)
        x1 = self.relu(x1)
        x1 = self.conv11(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.conv11(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x4 = self.conv11(x4)

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.se(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.avgpool(x)
        x2 = self.maxpool2(x)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), 1)
        x = self.head(x)
        c = torch.sigmoid(self.dense2(x))
        return x, c


def SEResNet(pretrained=False, num_classes=10):
    model = ResNetPlus3(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = SEResNet()
    # print(model.eval())

    # input_tensor = torch.randn(1, 12, 1)
    input_tensor = torch.randn(1, 1, 5250)
    print(model(input_tensor))


    # tensor([[0.2352, -0.1563, 0.3929, -0.4160, 0.5168, 0.1284, -0.0544, -0.1068,
    #          -0.9480, 0.1531, 0.4911, -0.1624, -0.3681, -0.2667, 0.2650, -0.0101,
    #          -0.2210, -0.4284]], grad_fn= < AddmmBackward >)
