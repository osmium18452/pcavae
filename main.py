import multiprocessing as mp
import os
import pickle
from DataLoader import DataLoader

import numpy as np


if __name__ == '__main__':
    data_dir = '/remote-home/liuwenbo/pycproj/tsdata/data'
    dataset = 'smd'
    map_dir = '/remote-home/liuwenbo/pycproj/tsdata/maps/smd'
    map = 'machine-1-1.camap.pkl'
    train_set_file = os.path.join(data_dir, dataset, 'train/machine-1-1.pkl')
    test_set_file = os.path.join(data_dir, dataset, 'test/machine-1-1.pkl')
    label_file = os.path.join(data_dir, dataset, 'label/machine-1-1.pkl')
    dataloader = DataLoader(train_set_file, test_set_file, label_file)
    dataloader.prepare_data(os.path.join(map_dir, map), cnn_window_size=5, vae_window_size=1)
    cnn_train_set = dataloader.load_cnn_train_set()
    cnn_test_set = dataloader.load_cnn_test_set()
    vae_train_set = dataloader.load_vae_train_set()
    vae_test_set = dataloader.load_vae_test_set()
    label_set = dataloader.load_label_set()
    vae_dim_list=dataloader.load_vae_dim_list()
    print(dataloader.load_vae_dim_list())
