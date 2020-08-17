import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn
# from main import device
from measures_optimized import GetIndicator
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DistanceAnalizer():
    def __init__(self, model):
        print('-----')

        self.layer_info_data = []
        self.model_modules = model.modules()
        self.ResetDebugInfo()
        self.his_neighbor_one_point = []
        for i in range(11):
            self.his_neighbor_one_point.append(
                torch.tensor([], dtype=int, device=device))

    def ResetDebugInfo(self,):
        self.layer_info_data = []

        for layer in range(7):
            self.layer_info_data.append(torch.tensor([], device=device))
        # self.layer_info_data.append(torch.tensor([], device=device))
        self.layer_info_data.append(torch.tensor([], device=device, dtype=int))

    def AppendInfo(self, data):

        for i, d in enumerate(data[:-1]):
            # print('--------------------->>>')
            # print(i, d.shape, self.layer_info_data[i].shape)
            self.layer_info_data[i] = torch.cat((self.layer_info_data[i], d))
        i += 1
        # print(i)

        self.layer_info_data[i] = torch.cat(
            (self.layer_info_data[i], data[i].long()))

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

    def SaveNeighbor(self, distance, path, name):
        s_, indices = torch.sort(distance, dim=1)
        # indices_np = indices.detach().cpu().numpy()
        # np.savetxt(path+'Neighbor'+name+'.csv',
        #            indices_np, fmt='%d', delimiter=',')

        for i in range(11):
            self.his_neighbor_one_point[i] = torch.cat(
                (self.his_neighbor_one_point[i], indices[i].view(1, -1)), dim=0)
            # print('--->>')
            # print(indices[i].shape, self.his_neighbor_one_point[i].shape)
            indices_np = self.his_neighbor_one_point[i].detach().cpu().numpy()
            np.savetxt(path+'Neighbor'+name[-5:]+str(i)+'.csv',
                       indices_np, fmt='%d', delimiter=',')

        return indices

    def ShowHistPairwiseDistanceAll(self, pairwist_distance, path, name=''):
        p = pairwist_distance.detach().cpu().numpy().reshape((-1, 1))
        # print(p.shape)
        plt.figure()
        plt.hist(p, bins=3000)

        axes = plt.gca()
        ylim = axes.get_ylim()
        xlim = axes.get_xlim()
        step = (ylim[1]-ylim[0])/30
        i = 2
        for key in self.indicator:
            plt.text(0.01+xlim[0], ylim[1]-i*step, s=str(key) +
                     '='+str(self.indicator[key])[:6])
            i += 1

        plt.savefig(path+'distance/hist_of_distance' +
                    name+'.png', dpi=300)
        plt.close()

    def ShowHeatmapPairwiseDistance(self, pairwist_distance, path, name=''):

        p = pairwist_distance.detach().cpu().numpy()
        plt.figure()
        heat = (p <= 1.e-6)
        seaborn.heatmap(heat)
        plt.savefig('pic/sphere/distance/heatmap_of_distance_all' +
                    name+'.png', dpi=300)
        plt.close()
        # p = pairwist_distance.detach().cpu().numpy()
        plt.figure()
        heat = (p)
        seaborn.heatmap(heat)
        plt.savefig(path +
                    '/distance/heatmap_of_distance_all' +
                    name+'.png', dpi=300)
        plt.close()

    def ShowHistPairwiseDistanceOne(self, pairwist_distance, path, name=''):

        for i in range(11):
            plt.subplot(6, 2, i+1)
            index = (self.label == i).nonzero()[0]
            # print(index)
            p = pairwist_distance[index].detach(
            ).cpu().numpy().reshape((-1, 1))
            plt.hist(p, bins=50)

        index = pairwist_distance.sum(dim=0).argmax()
        plt.subplot(6, 2, 12)
        p = pairwist_distance[index].detach().cpu().numpy().reshape((-1, 1))
        plt.hist(p, bins=50)
        plt.tight_layout()

        plt.savefig(path + '/distance/hist_of_distance_one' +
                    str(i) + name+'.png', dpi=300)
        plt.close()

    def AnalisisInfo(self, path, txt):

        s, index = torch.sort(self.layer_info_data[-1])
        # print(self.layer_info_data[-1].shape)
        self.label = self.layer_info_data[-1]
        data = [
            self.layer_info_data[0][index].to(device),
            self.layer_info_data[1][index].to(device),
            self.layer_info_data[2][index].to(device),
            self.layer_info_data[3][index].to(device),
        ]
        d_data = []

        for i in range(len(data)):
            name = '{}_layer:{}'.format(txt, i)
            # print(data[0].shape)
            # print(data[i].shape)
            self.indicator = GetIndicator(data[0],
                                          data[i])

            d_la = self.CalPairwiseDistance(data[i])
            # self.ShowHistPairwiseDistanceAll(d_la, path, name=name)
            # self.ShowHistPairwiseDistanceOne(d_la, path, name=name)
            # self.ShowHeatmapPairwiseDistance(d_la, path, name=name)

            d_data.append(d_la)

            # if i > 0:
            #     self.ShowHistPairwiseDistanceAll(
            #         d_data[i]-d_data[i-1], path,
            #         name='change_{}:{}'.format(i, i-1)+name)
            #     self.ShowHistPairwiseDistanceAll(
            #         d_data[i]-d_data[0], path,
            #         name='change_{}:{}'.format(i, 0)+name)
            if i == 3:
                self.SaveNeighbor(d_la, path, txt)
        min_n = d_la.min()
        # print(min_n)

        self.ResetDebugInfo()
