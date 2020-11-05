import numpy as np
import umap
import matplotlib.pyplot as plt
import main



def mainumP(dataname =  'MNIST'):
    pathdata =  'save{}data.npy'.format(dataname)
    pathlabel = 'save{}label.npy'.format(dataname)

    data = np.load(pathdata) 
    label = np.load(pathlabel)

    print(data.shape)

    t = umap.UMAP()
    dataout = t.fit_transform(data)

    # plt.figure(figsize=(5,5))

    fig, ax=plt.subplots(figsize=(5,5))
    # plt.title('UMAP', fontsize=32)
    plt.scatter(dataout[:,0], dataout[:,1], s=10, c=label,cmap='rainbow')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('{}.png'.format(dataname))


    indicator = main.CompPerformMetrics(
                    data.reshape(data.shape[0], -1), 
                    dataout,
                    dataset = dataname
                )
    
    print(indicator)
    print(dataname, indicator, file=open('saveindicator.txt', 'a'))

if __name__ == "__main__":
    mainumP(dataname =  'MNIST')
    mainumP(dataname =  'SwissRoll_')
    mainumP(dataname =  'Spheres5500')
    mainumP(dataname =  'Spheres10000')
