import torch
import numpy as np
from sklearn.datasets import make_s_curve
from samples_generator_new import make_swiss_roll

# shuffling the data to get an index of the individual batches
class SampleIndexGenerater():
    def __init__(self, data, batch_size):

        self.num_train_sample = data.shape[0]
        self.batch_size = batch_size
        self.Reset()

    def Reset(self):

        self.unuse_index = torch.randperm(self.num_train_sample).tolist()

    def CalSampleIndex(self, batch_idx):

        use_index = self.unuse_index[:self.batch_size]
        self.unuse_index = self.unuse_index[self.batch_size:]

        return use_index
     
        
def LoadData(data_name='SwissRoll', data_num=1500, seed=0, noise=0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), remove=None):

    """
    function used to load data

    Arguments:
        data_name {str} -- the dataset to be loaded
        data_num {int} -- the data number to be loaded
        seed {int} -- the seed for data generation
        noise {float} -- the noise for data generation
        device {torch} -- the device to store data
        remove {str} -- Shape of the points removed from the generated manifold
    """

    # Load SwissRoll Dataset
    if data_name == 'SwissRoll':
        if remove is None:
            train_data, train_label = make_swiss_roll(n_samples=data_num, noise=noise, random_state=seed)
        else:
            train_data, train_label = make_swiss_roll(n_samples=data_num, noise=noise, random_state=seed+1, remove=remove, center=[10, 10], r=8)
        train_data = train_data / 20

    # Load SCurve Dataset
    if data_name == 'SCurve':
        train_data, train_label = make_s_curve(n_samples=data_num, noise=noise, random_state=seed)
        train_data = train_data / 2

    # Put the data to device
    train_data = torch.tensor(train_data).to(device)
    train_label = torch.tensor(train_label).to(device)

    return train_data, train_label