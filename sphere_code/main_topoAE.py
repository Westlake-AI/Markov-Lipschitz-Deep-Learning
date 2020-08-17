import argparse
# import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import optim
# from torch.nn import functional as F
import model
import model_topoae
# from visdomtool import viz
# import visdomtool
import dataset
import datetime
import baseline
from measures_optimized import GetIndicator
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time

from Analizer import DistanceAnalizer

parser = argparse.ArgumentParser(description="author: Zelin Zang")
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def GetMaxLength(num_labels, train_label):
    maxle = 0
    for i in range(num_labels):
        length = torch.sum(train_label == i)
        # print(length)
        if maxle < length:
            maxle = length
    return maxle


def train(
        model, epoch, train_data, train_label,
        batch_size):
    model.train()
    train_loss = 0

    num_train_sample = train_data.shape[0]

    rand_index = torch.randperm(num_train_sample)
    # print(rand_index)

    num_batch = num_train_sample//batch_size
    for batch_idx in torch.arange(0, num_batch):

        # Prepare data and label
        start_number = (batch_idx * batch_size).int()
        end_number = torch.min(torch.tensor(
            [batch_idx*batch_size+batch_size, num_train_sample])).int()

        sample_index = rand_index[start_number:end_number]
        data = train_data[sample_index].float()
        label = train_label[sample_index]

        # add data to device
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)

        # forward
        _ = model(data, label)
        loss = model.Loss()
        loss.backward()

        cur_loss = loss.item()
        train_loss += cur_loss

        optimizer.step()
        if batch_idx % (param['LOGINTERVAL'] * num_batch) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_train_sample,
                batch_size*100.*batch_idx / num_train_sample,
                cur_loss/len(data)))

    # print(train_loss)
    time_stamp = datetime.datetime.now()
    print("time_stamp:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_train_sample
    ))
    model.PrintLoss()


def PlotLatenSpace(model, batch_size, device, datas, labels, da=0, txt='no name'):
    model.train()
    # model.debug = param['Debuginfo']
    # model.ResetDebugInfo()

    # train_loss = 0

    num_train_sample = datas.shape[0]
    # rand_index = torch.randint(
    #     low=0, high=num_train_sample, size=(num_train_sample,))

    for batch_idx in torch.arange(0, num_train_sample//batch_size+1):
        start_number = (batch_idx * batch_size).int()
        end_number = torch.min(torch.tensor(
            [batch_idx*batch_size+batch_size, num_train_sample])).int()

        # sample_index = datas[start_number:end_number]

        data = datas[start_number:end_number].float()
        label = labels[start_number:end_number]

        data = data.to(device)
        label = label.to(device)

        _ = model(data, label)

        if batch_idx == 0:
            latent_point = model.GetLatent()
            label_point = label.cpu().detach().numpy()
        else:
            latent_point = np.concatenate(
                (latent_point, model.GetLatent()), axis=0)
            label_point = np.concatenate(
                (label_point, label.cpu().detach().numpy()), axis=0)
        # print(end_number)
        if param['Debuginfo']:
            data_info = model.CalInfoDistance()
            da.AppendInfo(data_info)

    indicator = GetIndicator(datas.reshape(
        datas.shape[0], -1), torch.tensor(latent_point, device=device))

    plt.figure()
    clist = ['r', 'g', 'b', 'y', 'm', 'c', 'k',
             'pink', 'lightblue', 'lightgreen', 'grey']
    num_labels = int(labels.max())+1
    for ii in range(num_labels):
        i = num_labels - ii - 1
        point = latent_point[label_point == i]
        plt.scatter(point[:, 0], point[:, 1], c=clist[i], s=0.4,)

    axes = plt.gca()
    ylim = axes.get_ylim()
    xlim = axes.get_xlim()
    step = (ylim[1]-ylim[0])/30
    i = 1
    for key in indicator:
        plt.text(xlim[0], ylim[1]-i*step, s=str(key) +
                 '='+str(indicator[key])[:6])
        i += 1

    plt.title(txt)
    filename = 'M={}_K={}_E={}_loss={}_ML={}_SSS={}_TOPOAE:{}'.format(
        param['BATCHSIZE'], param['MAEK'], param['EPOCHS'],
        param['RATIO'], param['MultiLayerLoss'], param['SSS'], param['TopoAE'])
    path = 'pic/'+param['DATASET']+'/'+filename+txt+'.png'

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    if param['Debuginfo']:
        path = 'pic/'+param['DATASET']+'/'
        da.AnalisisInfo(path, filename+txt)
    # visdomtool.ShowImg(path, model.model_name+name)


# --- main function --- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--DATASET", default='sphere',
                        type=str, choices=[
                            'sphere', 'sphere5500', 'sphere55000',
                            'mnist', 'Fmnist',
                        ])
    parser.add_argument("-R", "--LEARNINGRATE", default=1e-4, type=float)
    parser.add_argument("-B", "--BATCHSIZE", default=64, type=int)
    parser.add_argument("-K", "--MAEK", default=25, type=int)
    parser.add_argument("-Z", "--ZDIM", default=2, type=int)
    parser.add_argument("-L", "--LOGINTERVAL", default=1.0, type=float)
    parser.add_argument(
        "-Ri", "--RATIO", default=[1.0, 1.0, 0], type=float)
    parser.add_argument("-E", "--EPOCHS", default=1000, type=int)
    parser.add_argument("-P", "--PlotForloop", default=200, type=int)
    parser.add_argument("-S", "--SSS", default=0, type=int)
    parser.add_argument("-ML", "--MultiLayerLoss", default=True, type=bool)
    parser.add_argument("-DI", "--Debuginfo", default=False, type=bool)
    parser.add_argument("-L2N", "--Loss2Norm", default=False, type=bool)
    parser.add_argument("-T", "--TopoAE", default=True, type=bool)

    args = parser.parse_args()
    param = args.__dict__
    for v, k in param.items():
        print('{v}:{k}'.format(v=v, k=k))

    train_data, train_label, test_data, test_label = dataset.LoadData(
        dataname=param['DATASET'])

    In_DIM = train_data.view(train_data.shape[0], -1).shape[1]
    # Model = model.MAE2(param['ZDIM'], In_DIM=In_DIM, param=param).to(device)
    Model = model_topoae.TopologicallyRegularizedAutoencoder().to(device)
    optimizer = optim.Adam(Model.parameters(), lr=param['LEARNINGRATE'])

    print(train_data.shape)

    # if param['Debuginfo']:
    da_tr = DistanceAnalizer(Model)
    da_te = DistanceAnalizer(Model)

    for epoch in range(1, param['EPOCHS'] + 1):

        if epoch == 1:
            Model.mae = True

        train(Model, epoch, train_data, train_label,
              batch_size=param['BATCHSIZE'],)
        # print(test_label)

        if epoch > 0 and epoch % param['PlotForloop'] == 0:
            name = 'epoch:' + str(epoch)
            PlotLatenSpace(Model, param['BATCHSIZE'], device,
                           test_data, test_label, da_tr, txt=name+'test')
            # PlotLatenSpace(Model, param['BATCHSIZE'], device,
            #                train_data, train_label, da_te, txt=name+'train')
