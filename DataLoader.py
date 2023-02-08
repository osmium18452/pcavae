import pickle

import numpy as np


class DataLoader:
    def __init__(self, train_set_file, test_set_file, label_file, window_size=1):
        f = open(train_set_file, 'rb')
        self.train_set: np.ndarray
        self.train_set = pickle.load(f)
        f.close()

        f = open(test_set_file, 'rb')
        self.test_set: np.ndarray
        self.test_set = pickle.load(f)
        f.close()

        f = open(label_file, 'rb')
        self.labels: np.ndarray
        self.labels = pickle.load(f)
        f.close()

        print(self.train_set.shape,self.test_set.shape,self.labels.shape)


if __name__ == '__main__':
    data_dir = '/remote-home/liuwenbo/pycproj/tsdata/data'
    dataset = 'smd'
    train_set_file = 'train/machine-1-1.pkl'
    test_set_file = 'test/machine-1-1.pkl'
    label_file = 'label/machine-1-1.pkl'
    dataloader = DataLoader(train_set_file,test_set_file,label_file)

