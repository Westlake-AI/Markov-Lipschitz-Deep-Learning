import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image

from measures_optimized import GetIndicator
from main import device

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clist = ['r', 'g', 'b', 'y', 'm', 'c', 'k',
         'pink', 'lightblue', 'lightgreen', 'r']


class AE(nn.Module):
    def __init__(self, Z_DIM, In_DIM=784):
        super().__init__()
        self.fc1 = nn.Linear(In_DIM, In_DIM//2)
        self.fc2 = nn.Linear(In_DIM//2, In_DIM//4)

        self.fcm = nn.Linear(In_DIM//4, Z_DIM)

        self.fc3 = nn.Linear(Z_DIM, In_DIM//4)
        self.fc4 = nn.Linear(In_DIM//4, In_DIM//2)
        self.fc5 = nn.Linear(In_DIM//2, In_DIM)

        self.model_name = 'AE'

    def Encoder(self, x):

        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        lat = self.fcm(h2)

        self.x_1 = x
        self.x_2 = h1
        self.x_3 = h2

        return lat

    def Decoder(self, lat):
        h3 = F.relu(self.fc3(lat))
        h4 = F.relu(self.fc4(h3))
        out = self.fc5(h4)

        self.hatx_1 = out
        self.hatx_2 = h4
        self.hatx_3 = h3

        return out

    def GetLatent(self):
        return self.lat.detach().numpy()

    def forward(self, x1, label):
        # x: [batch size, 1, 28,28] -> x: [batch size, 784]
        x1 = x1.view(-1, 784)
        self.lat = self.Encoder(x1)
        return self.Decoder(self.lat)  # , class_out

    def Loss(self,):
        loss_function = torch.nn.MSELoss()

        l = loss_function(self.hatx_1, self.x_1)

        return l


class MAE1(nn.Module):
    def __init__(self, Z_DIM, In_DIM=784):
        super().__init__()
        self.fc1 = nn.Linear(In_DIM, In_DIM//2)
        self.fc2 = nn.Linear(In_DIM//2, In_DIM//4)

        self.fcm = nn.Linear(In_DIM//4, Z_DIM)

        self.fc3 = nn.Linear(Z_DIM, In_DIM//4)
        self.fc4 = nn.Linear(In_DIM//4, In_DIM//2)
        self.fc5 = nn.Linear(In_DIM//2, In_DIM)

        self.model_name = 'MAE1_reconstraction_loss'

    def Encoder(self, x):

        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        lat = self.fcm(h2)

        self.x_1 = x
        self.x_2 = h1
        self.x_3 = h2

        return lat

    def Decoder(self, lat):
        h3 = F.relu(self.fc3(lat))
        h4 = F.relu(self.fc4(h3))
        out = self.fc5(h4)

        self.hatx_1 = out
        self.hatx_2 = h4
        self.hatx_3 = h3

        return out

    def GetLatent(self):
        return self.lat.cpu().detach().numpy()

    def forward(self, x1, label):
        # x: [batch size, 1, 28,28] -> x: [batch size, 784]
        x1 = x1.view(x1.shape[0], -1)
        # x1 = x1.view(-1, 784)
        self.lat = self.Encoder(x1)
        return self.Decoder(self.lat)  # , class_out

    def Loss(self,):
        loss_function = torch.nn.MSELoss()

        l = loss_function(self.hatx_1, self.x_1)

        return l


class MAE2(nn.Module):
    def __init__(self, Z_DIM,
                 In_DIM=784, k=5, param=None
                 ):
        super().__init__()
        self.fc1 = nn.Linear(In_DIM, In_DIM*7//8)
        self.fc2 = nn.Linear(In_DIM*7//8, In_DIM*6//8)

        self.fcm = nn.Linear(In_DIM*6//8, Z_DIM)

        self.fc3 = nn.Linear(Z_DIM, In_DIM*6//8)
        self.fc4 = nn.Linear(In_DIM*6//8, In_DIM*7//8)
        self.fc5 = nn.Linear(In_DIM*7//8, In_DIM)

        self.mae = False
        self.loss3 = True
        self.RATIO = param['RATIO']

        self.model_name = 'MAE2_knn'
        self.k = param['MAEK']
        self.MultiLayerLoss = param['MultiLayerLoss']
        self.SSS = param['SSS']
        self.debug = False
        self.param = param
        self.Loss2Norm = param['Loss2Norm']

        if self.debug:
            self.ResetDebugInfo()

    def Encoder(self, x):

        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        lat = self.fcm(h2)

        self.x_1 = x
        self.x_2 = h1
        self.x_3 = h2

        return lat

    def Decoder(self, lat):
        h3 = F.leaky_relu(self.fc3(lat))
        h4 = F.leaky_relu(self.fc4(h3))
        out = self.fc5(h4)

        self.hatx_1 = out
        self.hatx_2 = h4
        self.hatx_3 = h3

        return out
    
    def ShowkNNGraph(self, data, kNN_mask, label):
        import matplotlib.pyplot as plt

        label_numpy = label.detach().numpy()
        data_numpy = data.detach().numpy()

        batchsize = kNN_mask.shape[0]

        for i in range(10):
            point = data_numpy[label_numpy == i]
            plt.scatter(point[:, 0], point[:, 1], c=clist[i], s=10)

        for i in range(batchsize):
            for j in range(batchsize):
                if kNN_mask[i, j] == 1:
                    plt.plot([data[i, 0], data[j, 0]], [
                             data[i, 1], data[j, 1]], 'grey', linewidth=0.5)

        plt.legend()
        plt.show()

    def kNNGraph(self, data):

        # import time
        k = self.k
        batch_size = data.shape[0]

        x = data
        y = data
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-12).sqrt()  # for numerical stabili

        kNN_mask = np.zeros((batch_size, batch_size))
        s_, indices = torch.sort(d, dim=1)
        indices = indices.detach().cpu().numpy()
        for i in range(batch_size):
            for j in range(k):
                kNN_mask[i, indices[i, j+1]] = 1

        kNN_mask = torch.tensor(kNN_mask).to(device)
        # self.neighbor = indices
        return d, kNN_mask

    def Decoder(self, lat):
        h3 = F.leaky_relu(self.fc3(lat))
        h4 = F.leaky_relu(self.fc4(h3))
        out = self.fc5(h4)

        self.hatx_1 = out
        self.hatx_2 = h4
        self.hatx_3 = h3

        return out

    def GetLatent(self):
        return self.lat.cpu().detach().numpy()

    def PrintLoss(self,):
        print('lne:{} lmse {} loss3 {}'.format(
            self.loss_r, self.loss_ne, self.loss_3))

    def OutDDatap1(self,):
        return self.d_datap1.cpu().detach().numpy()

    def Loss(self,):
        loss_mse = torch.nn.MSELoss()

        def loss_ne(data, datap1):
            d_data, kNN_mask_data = self.kNNGraph(data)
            d_datap1, kNN_mask_datap1 = self.kNNGraph(datap1)
            mask = kNN_mask_data + kNN_mask_datap1
            mask[mask == 2] = 1

            self.d_datap1 = d_datap1

            # loss2 = torch.norm(d_data.mul(mask) - d_datap1.mul(mask))/10000
            if self.Loss2Norm:
                norml_data = torch.sqrt(torch.tensor(float(data.shape[1])))
                norml_datap1 = torch.sqrt(torch.tensor(float(datap1.shape[1])))
                loss2 = torch.norm(
                    d_data.mul(mask)/norml_data -
                    d_datap1.mul(mask)/norml_datap1
                )/torch.sum(mask)
            else:
                loss2 = torch.norm(
                    d_data.mul(mask) - d_datap1.mul(mask)
                )/torch.sum(mask)

            if self.loss3:
                deta_d = 0.01 - d_datap1
                deta_d[d_data < 0.1] = 0
                deta_d[deta_d < 0] = 0

                loss3 = deta_d.sum()
            else:
                loss3 = 0

            return loss2, loss3

        lmse = loss_mse(self.hatx_1, self.x_1)
        if self.mae:
            if self.MultiLayerLoss:
                loss2_0, loss3_0 = loss_ne(self.x_1, self.lat)
                loss2_1, loss3_1 = loss_ne(self.x_1, self.x_2)
                loss2_2, loss3_2 = loss_ne(self.x_2, self.x_3)
                loss2_3, loss3_3 = loss_ne(self.x_3, self.lat)

                if self.SSS == 0:
                    loss2 = loss2_0
                    loss3 = loss3_0
                if self.SSS == 1:
                    loss2 = (loss2_1+loss2_2+loss2_3)/3
                    loss3 = (loss3_1+loss3_2+loss3_3)/3
                if self.SSS == 2:
                    loss2 = (5*loss2_0+8*loss2_1+10*loss2_2)/23
                    loss3 = (5*loss3_0+8*loss3_1+10*loss3_2)/23
                if self.SSS == 3:
                    loss2 = (loss2_0+loss2_1+loss2_2+loss2_3)/4
                    loss3 = (loss3_0+loss3_1+loss3_2+loss3_3)/4
                if self.SSS == 4:
                    loss2 = (5*loss2_0+8*loss2_1+10*loss2_2+10*loss2_3)/33
                    loss3 = (5*loss3_0+8*loss3_1+10*loss3_2+10*loss3_3)/33
                if self.SSS == 5:
                    loss2 = (loss2_1+loss2_2)/4
                    loss3 = (loss3_1+loss3_2)/4
            else:
                loss2, loss3 = loss_ne(self.x_1, self.lat)
        else:
            loss2, loss3 = 0, 0

        self.loss_r = self.RATIO[0] * lmse
        self.loss_ne = self.RATIO[1] * loss2
        self.loss_3 = self.RATIO[2] * loss3
        # print(lmse, loss2, loss3)

        loss = self.RATIO[0] * lmse + self.RATIO[1] * \
            loss2 + self.RATIO[2] * loss3
        return loss

    def CalInfoDistance(self):
        bag = []
        bag.append(self.x_1)
        bag.append(self.x_2)
        bag.append(self.x_3)
        bag.append(self.lat)
        bag.append(self.hatx_3)
        bag.append(self.hatx_2)
        bag.append(self.hatx_1)
        bag.append(self.label.long())
        return bag

    def CalInfoNeighbor(self):
        bag = []

        return bag

    def CalPairwiseDistance(self, data):
        x = data.float()
        y = data.float()
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-12).sqrt()  # for numerical stabili
        return d

    def forward(self, x1, label):
        x1 = x1.view(x1.shape[0], -1)
        self.lat = self.Encoder(x1)

        out = self.Decoder(self.lat)

        self.label = label

        return out  # , class_out
