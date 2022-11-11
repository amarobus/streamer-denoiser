import random
import numpy as np
import glob
import h5py
import tensorflow.keras as keras
import scipy.constants as co
seed = 2


# Reference: https://github.com/keras-team/keras/blob/v2.7.0/keras/utils/data_utils.py#L411-L486
class DataGenerator(keras.utils.Sequence):
    def __init__(self,  paths, batch_size = 4, dim = (7936, 1536), mask = (2304, None, None, None), n_channels = 1, shuffle = True, valid = False, test = False, clip = False):

        random.seed(seed)
        np.random.seed(seed)

        # Get file list
        original_files = sorted(glob.glob(paths[0] + '*.hdf'))
        noisy_files = sorted(glob.glob(paths[1] + '*.hdf'))

        num_samples = len(original_files)
        #shuffle list
        temp = list(zip(noisy_files, original_files))
        random.shuffle(temp)
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        noisy_files, original_files = list(res1), list(res2)

        # Data split
        # 20%
        if valid:
            noisy_files = noisy_files[int(0.7*num_samples):int(0.9*num_samples)]
            original_files = original_files[int(0.7*num_samples):int(0.9*num_samples)]
        # 10%
        elif test:
            noisy_files = noisy_files[int(0.9*num_samples):]
            original_files = original_files[int(0.9*num_samples):]
        # 70%
        else:
            noisy_files = noisy_files[:int(0.7*num_samples)]
            original_files = original_files[:int(0.7*num_samples)]

        self.batch_size = batch_size
        self.dim = dim
        self.mask = mask
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.noisy_files = noisy_files
        self.original_files = original_files
        self.clip = clip
        self.r = None
        # Shuffle indexes
        self.on_epoch_end()
        # set up mesh information
        self.read_data(original_files[0])


    def read_data(self, file):
        with h5py.File(file, 'r') as f:
            group = f['group1']
            data = np.array(group['charge_density'], dtype=np.float32)

            if self.r is None:
                self.dr = group.attrs['y.upper'] / group.attrs['y.num_cells']
                self.dz = group.attrs['x.upper'] / group.attrs['x.num_cells']
                self.rmax = group.attrs['y.upper']
                self.r = np.arange(0 + self.dr/2., self.rmax, self.dr)[None,...][None,...,None]

        if self.clip:
            return np.clip(data,-1,1)
        else:
            return data


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.original_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __len__(self):
        # Number of batches per epoch
        # A common practice is to set this value to [# samples / batch size] so that
        # the model sees the training samples at most once per epoch

        return int(np.floor(len(self.original_files) / self.batch_size))


    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # List of files
        noisy_files_temp = [self.noisy_files[k] for k in indexes]
        original_files_temp = [self.original_files[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(noisy_files_temp, original_files_temp)


        return X, y

    def __data_generation(self, noisy_files_temp, original_files_temp):

        # (n_samples, *dim, n_channels)
        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Y = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        mask = self.mask
        if mask:
            for i, noisy_file in enumerate(noisy_files_temp):
                X[i] = self.read_data(noisy_file)[mask[0]:mask[1],mask[2]:mask[3]][...,None]
                Y[i] = self.read_data(original_files_temp[i])[mask[0]:mask[1],mask[2]:mask[3]][...,None]
        else:
            for i, noisy_file in enumerate(noisy_files_temp):
                X[i] = self.read_data(noisy_file)[...,None]
                Y[i] = self.read_data(original_files_temp[i])[...,None]
        

        return X, Y
