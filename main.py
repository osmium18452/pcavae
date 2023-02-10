import argparse
import multiprocessing as mp
import os
import pickle

import torch
from torch import optim

from DataLoader import DataLoader
from VAE import VAE
from IVAE import IVAE

import numpy as np


def get_device_list(gpu_devices, vae_num):
    ret = []
    li = gpu_devices.strip().split(',')
    for i in li:
        ret.append(int(i))
    # print(ret,'ret')
    return np.tile(np.array(ret), vae_num)[:vae_num]


def train_vae(model, data, optimizer, device, train_set_size):
    train_set_size = data.shape[0]
    print(train_set_size)


def shuffle_data(data, size):
    shuffle = np.random.permutation(size)
    data = data[shuffle]
    print('data', data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', "--latent", default=5, type=int)
    parser.add_argument('-G', "--gpu", action="store_true")
    parser.add_argument("-r", "--learning_rate", default=0.01, type=float)
    parser.add_argument("-e", "--epoch", default=1000, type=int)
    parser.add_argument("-b", "--batch_size", default=1024, type=int)
    parser.add_argument("-w", "--window_size", default=20, type=int)
    parser.add_argument("-g", "--gpu_device", default="0", type=str)
    parser.add_argument('-N', '--normalize_data', action='store_true')
    args = parser.parse_args()

    latent_size = args.latent
    gpu = args.gpu
    learning_rate = args.learning_rate
    epoch = args.epoch
    batch_size = args.batch_size
    window_size = args.window_size
    gpu_device = args.gpu_device
    normalize = args.normalize_data

    data_dir = '/remote-home/liuwenbo/pycproj/tsdata/data'
    dataset = 'smd'
    map_dir = '/remote-home/liuwenbo/pycproj/tsdata/maps/smd'
    map = 'machine-1-1.camap.pkl'
    train_set_file = os.path.join(data_dir, dataset, 'train/machine-1-1.pkl')
    test_set_file = os.path.join(data_dir, dataset, 'test/machine-1-1.pkl')
    label_file = os.path.join(data_dir, dataset, 'label/machine-1-1.pkl')
    map_file = os.path.join(map_dir, map)

    ivae = IVAE(train_set_file, test_set_file, label_file, map_file, latent_size, gpu, learning_rate, window_size,
                gpu_device, normalize)
    ivae.train_simultaneously(epoch,batch_size,gpu)
