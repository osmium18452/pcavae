import argparse
import multiprocessing as mp
import os
import pickle

import torch
from torch import optim

from DataLoader import DataLoader
from VAE import VAE

import numpy as np


def get_device_list(gpu_devices, vae_num):
    ret = []
    li = gpu_devices.strip().split(',')
    for i in li:
        ret.append(int(i))
    # print(ret,'ret')
    return np.tile(np.array(ret), vae_num)[:vae_num]


def train_vae(model, data, optimizer, device,train_set_size):
    train_set_size=data.shape[0]
    print(train_set_size)

def shuffle_data(data,size):
    shuffle=np.random.permutation(size)
    data=data[shuffle]
    print('data',data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', "--latent", default=5, type=int)
    parser.add_argument('-G', "--gpu", action="store_true")
    parser.add_argument("-r", "--learning_rate", default=0.001, type=float)
    parser.add_argument("-e", "--epoch", default=10, type=int)
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
    dataloader = DataLoader(train_set_file, test_set_file, label_file, normalize=normalize)
    dataloader.prepare_data(os.path.join(map_dir, map), cnn_window_size=window_size, vae_window_size=1)

    train_set_size=dataloader.load_train_set_size()
    # load data
    cnn_train_set_x, cnn_train_set_y = dataloader.load_cnn_train_set()
    cnn_test_set_x, cnn_test_set_y = dataloader.load_cnn_test_set()
    vae_train_set = dataloader.load_vae_train_set()
    vae_test_set = dataloader.load_vae_test_set()
    label_set = dataloader.load_label_set()

    vae_dim_list = dataloader.load_vae_dim_list()

    # convert to torch.Tensor()
    cnn_train_set_x = torch.Tensor(cnn_train_set_x)
    cnn_train_set_y = torch.Tensor(cnn_train_set_y)
    cnn_test_set_x = torch.Tensor(cnn_test_set_x)
    cnn_test_set_y = torch.Tensor(cnn_test_set_y)
    for i in range(len(vae_train_set)):
        vae_train_set[i] = torch.squeeze(torch.Tensor(vae_train_set[i]))
        vae_test_set[i] = torch.squeeze(torch.Tensor(vae_test_set[i]))
    # print(vae_train_set[0].shape)

    # print(dataloader.load_vae_dim_list())
    # print('cnn x', cnn_train_set_x.shape, cnn_test_set_x.shape)
    # print('cnn y', cnn_train_set_y.shape, cnn_test_set_y.shape)
    # print('label', label_set.shape)
    # for i in range(len(vae_train_set)):
    #     print('vae', type(vae_train_set[i]), vae_test_set[i].shape)

    # create vaes
    vae_list = []
    optimizer_list = []
    device_list = get_device_list(gpu_device, vae_num=vae_dim_list.shape[0])
    vae_num=vae_dim_list.shape[0]
    print('device list',device_list,gpu_device)
    print(vae_dim_list)
    for i, size in enumerate(vae_dim_list):
        vae_list.append(VAE(size, latent_size, i))
        if gpu:
            vae_list[i].cuda(device=device_list[i])
        optimizer_list.append(optim.Adam(vae_list[i].parameters(), lr=learning_rate))

    # test train

    for i in range(epoch):
        if i%5==0:
            shuffle=np.random.permutation(train_set_size)
            # print(vae_train_set[0])
            for i in range(vae_num):
                vae_train_set[i]=vae_train_set[i][shuffle]
            # print(vae_train_set[0])
