import os
from time import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from CellDataset import *
from models.SEResNet import SEResNet
from utils import *


class MultiLabelCircleLoss(nn.Module):

    def __init__(self, reduction="mean", inf=1e12):
        super(MultiLabelCircleLoss, self).__init__()
        self.reduction = reduction
        self.inf = inf

    def forward(self, logits, labels):
        logits = (1 - 2 * labels) * logits  # <3, 4>
        logits_neg = logits - labels * self.inf  # <3, 4>
        logits_pos = logits - (1 - labels) * self.inf  # <3, 4>
        zeros = torch.zeros_like(logits[..., :1])  # <3, 1>
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)  # <3, 5>
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)  # <3, 5>
        neg_loss = torch.logsumexp(logits_neg, dim=-1)  # <3, >
        pos_loss = torch.logsumexp(logits_pos, dim=-1)  # <3, >
        loss = neg_loss + pos_loss
        if "mean" == self.reduction:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Model():

    def __init__(self, num_works=1, batch_size=8, epoch=500, train_mod=False):
        if train_mod:
            self.dataset = CellMemoryDataset()
        else:
            self.dataset = CellDiskDataset()
        self.num_classes = self.dataset.types_num
        net = SEResNet(num_classes=self.dataset.types_num)
        print(net.eval())

        self.root_path = './all/model_results/'
        self.model_save_path = self.root_path
        self.figure_save_path = self.root_path
        self.board_writer_path = f'{self.root_path}tensorboard/'
        self.checkpoint_path = f'{self.root_path}checkpoints/'

        if not os.path.isdir(self.root_path):
            os.makedirs(self.root_path)
        if not os.path.isdir(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.isdir(self.figure_save_path):
            os.makedirs(self.figure_save_path)
        if not os.path.isdir(self.board_writer_path):
            os.makedirs(self.board_writer_path)
        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        # initialize tensorboard
        self.writer = SummaryWriter(self.board_writer_path)

        # create logger
        self.logger_path = f'{self.root_path}SERCNet.log'
        self.logger = creat_log("SERCNet", f"{self.logger_path}")
        self.logger.info("Initializing SERCNet...")

        # initialize the training parameters
        self.num_works = num_works
        self.batch_size = batch_size
        self.trainset_rate = 0.7  # the proportion of train set to total data set
        self.validset_rate = 0.3  # the proportion of valid set to total data set
        # self.testset_rate = 0.2  # the proportion of test set to total data set
        self.seed = 42
        self.lr = 0.001
        self.lr_last = 0.001
        self.start_epoch = -1
        self.EPOCH = epoch
        self.warm_up = 5
        self.patience = 10
        self.budget = 0.6  # budget of confidence loss
        self.lmbda = 0.1  # initial coefficient of confidence loss

        self.device_ids = [0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.net = torch.nn.DataParallel(self.net, device_ids=self.device_ids)
        self.optimizer = AdamW(self.net.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.EPOCH, eta_min=5e-6)
        self.loss_fn = MultiLabelCircleLoss()
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.BCEWithLogitsLoss()


    def fit(self):

        # initialize
        min_loss = np.inf
        list_of_train_loss, list_of_train_acc = [], []
        list_of_valid_loss, list_of_valid_acc = [], []


        # lengths = [round(len(self.dataset) * self.trainset_rate), len(self.dataset) - round(len(self.dataset) * self.trainset_rate)]

        # 随机划分数据集
        # train_set, valid_set = random_split(self.dataset, lengths)
        train_set = self.dataset
        # valid_set = self.dataset
        train_loader = DataLoader(train_set,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_works,
                                  pin_memory=True)
        # valid_loader = DataLoader(valid_set,
        #                           batch_size=self.batch_size,
        #                           shuffle=True,
        #                           num_workers=self.num_works,
        #                           pin_memory=True)

        for epoch in range(self.start_epoch + 1, self.EPOCH):

            t1 = time()

            train_loss, train_acc, train_conf = self.train(train_loader)
            # valid_loss, valid_acc, valid_conf = self.valid(valid_loader)

            t2 = time()

            self.logger.info(f"epoch {epoch}: train_loss {train_loss:.4f}; "
                             f"train_acc {train_acc:.4f}; "
                             f"train_conf {train_conf:.4f}; "
                             f"time {(t2 - t1):.1f} s")

            # save the output of each epoch
            list_of_train_loss.append(train_loss)
            list_of_train_acc.append(train_acc)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)

            # save the best model
            if (train_loss < min_loss) and (epoch > self.warm_up):
                min_loss = train_loss
                checkpoint = {
                    "net": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }

                self.logger.info(f"best score: {min_loss:.4f} @ {epoch}")
                torch.save(checkpoint, f'{self.checkpoint_path}ckpt_best.pth')

        # save the final model
        torch.save(self.net.state_dict(), f"{self.model_save_path}model.pth")

        return 0

    def train(self, train_loader):

        self.net.train()

        epoch_loss = 0
        epoch_acc = 0
        epoch_conf = 0

        for xrd, label in tqdm(train_loader, desc='Training'):
            xrd = xrd.to(self.device)
            label = label.to(self.device)

            out, conf = self.net(xrd)
            pred_y = out.detach().argmax(dim=1)
            acc = (pred_y == label).float().mean()

            self.optimizer.zero_grad()
            one_hot_label = torch.nn.functional.one_hot(label.squeeze(), num_classes=self.num_classes).float()
            loss = self.loss_fn(out, one_hot_label)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_conf += conf.detach().cpu().numpy().mean()

        return epoch_loss / len(train_loader), epoch_acc / len(train_loader), epoch_conf / len(train_loader)



if __name__ == '__main__':
    model = Model(num_works=13, batch_size=128, epoch=500, train_mod=True)
    model.fit()
