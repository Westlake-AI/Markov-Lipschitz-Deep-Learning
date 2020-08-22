#import tadasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed
from torchvision import datasets, transforms
from mpl_toolkits.mplot3d import Axes3D
# from .custom_shapes import dsphere
# from visdomtool import viz


def dsphere(n=100, d=2, r=1, noise=None, ambient=None):
    """
    Sample `n` data points on a d-sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """
    data = np.random.randn(n, d+1)

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)

    return data


def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    # it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance = 10/np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 10*n_samples  # int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        # plt.savefig('look.png')
        # plt.show()

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    index_seed = np.linspace(0, 10000, num=20, dtype='int16', endpoint=False)
    arr = np.array([], dtype='int16')
    for i in range(500):
        arr = np.concatenate((arr, index_seed+int(i)))
    # arr.astype(int)
    print(arr.shape)

    # rng.shuffle(arr)
    dataset = dataset[arr]
    labels = labels[arr]

    return dataset/22+0.5, labels


def create_sphere_dataset5500(n_samples=1000, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    # it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance = 10/np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 1*n_samples  # int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        # plt.savefig('look.png')
        # plt.show()

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    # index_seed = np.linspace(
    #     0, dataset.shape[0], num=11, dtype='int16', endpoint=False)
    # arr = np.array([], dtype='int16')
    # for i in range(500):
    #     arr = np.concatenate((arr, index_seed+int(i)))
    # arr.astype(int)
    # print(arr.shape)
    arr = np.arange(dataset.shape[0])
    np.random.shuffle(arr)
    # rng.shuffle(arr)
    dataset = dataset[arr]
    labels = labels[arr]

    return dataset/22+0.5, labels


def create_sphere_dataset55000(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    # it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance = 10/np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 100*n_samples  # int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        # plt.savefig('look.png')
        # plt.show()

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    index_seed = np.linspace(
        0, dataset.shape[0], num=110, dtype='int16', endpoint=False)
    arr = np.array([], dtype='int16')
    for i in range(500):
        arr = np.concatenate((arr, index_seed+int(i)))
    # arr.astype(int)
    # print(arr.shape)

    # rng.shuffle(arr)
    dataset = dataset[arr]
    labels = labels[arr]

    return dataset/22+0.5, labels


def LoadData(
    dataname='mnist',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

):

    if dataname == 'mnist':

        train_data = datasets.MNIST(
            '~/data', train=True, download=True,
            transform=transforms.ToTensor()
        ).data.float()/255
        train_label = datasets.MNIST(
            '~/data', train=True, download=True,
            transform=transforms.ToTensor()
        ).targets

        test_data = datasets.MNIST(
            '~/data', train=False,
            transform=transforms.ToTensor()
        ).data.float()/255

        test_labels = datasets.MNIST(
            '~/data', train=False,
            transform=transforms.ToTensor()
        ).targets
        # print(train_data.max())
    if dataname == 'Fmnist':

        train_data = datasets.FashionMNIST(
            '~/data', train=True, download=True,
            transform=transforms.ToTensor()
        ).data.float()/255
        train_label = datasets.FashionMNIST(
            '~/data', train=True, download=True,
            transform=transforms.ToTensor()
        ).targets

        test_data = datasets.FashionMNIST(
            '~/data', train=False,
            transform=transforms.ToTensor()
        ).data.float()/255

        test_labels = datasets.FashionMNIST(
            '~/data', train=False,
            transform=transforms.ToTensor()
        ).targets
        print(train_data.max())

    if dataname == 'Spheres10000':
        train_data, train_label = create_sphere_dataset()
        test_data, test_labels = create_sphere_dataset()

        train_test_split = train_data.shape[0] * 9//10
        train_data = torch.tensor(train_data).to(device)[:7500]
        train_label = torch.tensor(train_label).to(device)[:7500]
        test_data = torch.tensor(test_data).to(device)[9000:]
        test_labels = torch.tensor(test_labels).to(device)[9000:]
    if dataname == 'Spheres5500':
        train_data, train_label = create_sphere_dataset5500(seed=42)
        test_data, test_labels = create_sphere_dataset5500(seed=42)

        train_test_split = train_data.shape[0] * 5//10
        train_data = torch.tensor(train_data).to(device)[:train_test_split]
        train_label = torch.tensor(train_label).to(device)[:train_test_split]
        test_data = torch.tensor(test_data).to(device)[train_test_split:]
        test_labels = torch.tensor(test_labels).to(device)[train_test_split:]
    if dataname == 'sphere55000':
        train_data, train_label = create_sphere_dataset55000()
        test_data, test_labels = create_sphere_dataset55000()
        # print(train_data.shape, train_label.shape)
        # input('-------')
        train_test_split = train_data.shape[0] * 9//10
        train_data = torch.tensor(train_data).to(device)[:train_test_split]
        train_label = torch.tensor(train_label).to(device)[:train_test_split]
        test_data = torch.tensor(test_data).to(device)[train_test_split:]
        test_labels = torch.tensor(test_labels).to(device)[train_test_split:]

    return train_data, train_label, test_data, test_labels


def create_sphere_dataset_test(n_samples=500, d=100, n_spheres=11, r=5,
                               plot=True, seed=42):
    np.random.seed(seed)

    # it seemed that rescaling the shift variance by sqrt of d lets big
    # sphere stay around the inner spheres
    variance = 10/np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 2*n_samples  # int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*7)
    spheres.append(big)
    n_datapoints += n_samples_big

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    if plot:
        print(dataset.shape)
        print(labels.shape)
        viz.scatter(
            X=dataset[:, 0:3],
            Y=labels+1,
            opts={
                'markersize': 5,
                'title': 'sphere',
                'legend': [
                    'manifold 0', 'manifold 1', 'manifold 2', 'manifold 3',
                    'manifold 4', 'manifold 5', 'manifold 6', 'manifold 7',
                    'manifold 8', 'manifold 9', 'manifold 10'
                ]
            },
        )
    return dataset/22+0.5, labels


if __name__ == "__main__":
    create_sphere_dataset_test()
