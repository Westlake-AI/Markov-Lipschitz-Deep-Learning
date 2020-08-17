

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.manifold import Isomap
from measures_optimized import MeasureCalculator
import numpy as np
import dataset


def PLotFig(X_embedded, label, indicator, name):

    plt.figure()
    clist = ['r', 'g', 'b', 'y', 'm', 'c', 'k',
             'pink', 'lightblue', 'lightgreen', 'grey']
    num_labels = int(label.max())+1
    for ii in range(num_labels):
        i = num_labels - ii - 1
        point = X_embedded[label == i]
        plt.scatter(point[:, 0], point[:, 1], c=clist[i], s=1,)

    axes = plt.gca()
    ylim = axes.get_ylim()
    xlim = axes.get_xlim()
    step = (ylim[1]-ylim[0])/30
    i = 2
    for key in indicator:
        plt.text(0.01+xlim[0], ylim[1]-i*step, s=str(key) +
                 '='+str(indicator[key])[:6])
        i += 1

    # name = txt
    # path = 'pic/'+ dataset_name+ '/' +model.model_name+name+'.png'
    plt.savefig('pic/'+'othermehtod/'+dn+'_'+name+'.png', dpi=300)
    plt.close()


def DimensionalityReduction(train, label, test_data, test_label, tool, name):

    train = train.detach().cpu().numpy().reshape((train.shape[0], -1))
    test_data = test_data.detach().cpu().numpy().reshape(
        (test_data.shape[0], -1))
    label = label.detach().cpu().numpy()
    test_label = test_label.detach().cpu().numpy()

    tool.fit(train)
    if name != 'tsne':
        X_embedded = tool.transform(test_data)
    else:
        X_embedded = tool.fit_transform(test_data)

    label = test_label
    indicator = GetIndicator(test_data, X_embedded)

    PLotFig(X_embedded, label, indicator, name)


def GetIndicator(real_data_i=None, latent_i=None):

    real_data = real_data_i
    latent = latent_i

    real_data = real_data-np.min(real_data)
    latent = latent-np.min(latent)
    real_data = real_data/np.max(real_data)
    latent = latent/np.max(latent)

    calc = MeasureCalculator(real_data, latent, 201)

    kl1 = calc.density_kl_global_1()
    kl01 = calc.density_kl_global_01()
    kl001 = calc.density_kl_global_001()

    mrreZX = []
    mrreXZ = []
    cont = []
    trust = []
    for k in range(10, 201, 10):
        # print('test k = {}'.format(k))
        mrreZX.append(calc.mrre(k)[0])
        mrreXZ.append(calc.mrre(k)[1])
        cont.append(calc.continuity(k))
        trust.append(calc.trustworthiness(k))

    rmse = calc.rmse()

    indicator = {}
    indicator['kl001'] = kl001
    indicator['kl01'] = kl01
    indicator['kl1'] = kl1
    indicator['mrre ZX'] = np.mean(mrreZX)
    indicator['mrre XZ'] = np.mean(mrreXZ)
    indicator['cont'] = np.mean(cont)
    indicator['trust'] = np.mean(trust)
    indicator['rmse'] = rmse

    print(indicator)
    # input('stop')

    return indicator


if __name__ == "__main__":

    dn = 'mnist'

    dimension = 2
    tool_tsne = TSNE(n_components=dimension)
    tool_umap = umap.UMAP(n_components=dimension)
    tool_isomap = Isomap(n_components=dimension)
    tool_pca = PCA(n_components=dimension)

    tool_list = [
        tool_pca,
        tool_umap,
        tool_isomap,
        tool_tsne,
    ]
    tool_name = [
        'pca',
        'umap',
        'isomap',
        'tsne',
    ]

    train_data, train_label, test_data, test_label = dataset.LoadData(
        dataname=dn)
    # print(train_data)
    for tool, name in zip(tool_list, tool_name):
        print(name)
        print('----------------------')
        DimensionalityReduction(train_data, train_label,
                                test_data, test_label, tool, name)
