import random
import numpy as np
import glob
import h5py
import tensorflow.keras as keras
import scipy.constants as co
seed = 2


# Reference: https://github.com/keras-team/keras/blob/v2.7.0/keras/utils/data_utils.py#L411-L486
class DataGenerator(keras.utils.Sequence):
    def __init__(self,  batch_size = 16, dim = (4096, 768), n_channels = 1, shuffle = True, valid = False, test = False, clip = True):

        random.seed(seed)
        np.random.seed(seed)

        original_dir = '/mnt/lustre/scratch/nlsas/home/csic/eli/ama/data/charge_density/original/'
        noisy_dir = '/mnt/lustre/scratch/nlsas/home/csic/eli/ama/data/charge_density/noisy/'

        # Get file list
        original_files = sorted(glob.glob(original_dir + '*.hdf'))
        noisy_files = sorted(glob.glob(noisy_dir + '*.hdf'))

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


        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.noisy_files = noisy_files
        self.original_files = original_files
        self.clip = clip
        # Shuffle indexes
        self.on_epoch_end()


    def read_data(self, file):
        with h5py.File(file, 'r') as f:
            group = f['group1']
            data = np.array(group['charge_density'], dtype=np.float32)

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

        for i, noisy_file in enumerate(noisy_files_temp):
            X[i] = self.read_data(noisy_file)[::2,::2][1024:][...,None]
            Y[i] = self.read_data(original_files_temp[i])[::2,::2][1024:][...,None]


        return X, Y
