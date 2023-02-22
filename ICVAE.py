import os
import time

import torch
from tqdm import tqdm
from torch import nn, optim
import numpy as np

from CVAE import CVAE
from DataLoader import DataLoader


class ICVAE:
    def __init__(self, dataloader, latent_size, gpu, learning_rate, gpu_device):
        self.dataloader = dataloader

        self.train_set_size = dataloader.load_train_set_size()
        self.test_set_size = dataloader.load_test_set_size()
        # load data
        train_input, train_condition = dataloader.load_cvae_train_data()
        test_input, test_condition = dataloader.load_cvae_test_data()
        label_set = dataloader.load_label_set()

        self.cvae_dim_list = dataloader.load_vae_dim_list()
        print('cvae dim list:', self.cvae_dim_list)

        self.label_set = torch.Tensor(label_set)
        self.train_input = []
        self.train_condition = []
        self.test_input = []
        self.test_condition = []
        self.validate_input = []
        self.validate_condition = []
        for i in range(len(train_input)):
            self.train_input.append(torch.squeeze(torch.Tensor(train_input[i])))
            self.train_condition.append(torch.squeeze(torch.Tensor(train_condition[i])))
            self.test_input.append(torch.squeeze(torch.Tensor(test_input[i])))
            self.test_condition.append(torch.squeeze(torch.Tensor(test_condition[i])))
            self.validate_input.append(torch.squeeze(torch.Tensor(self.test_input[-1][:100])))
            self.validate_condition.append(torch.squeeze(torch.Tensor(self.test_condition[-1][:100])))

        # create cvaes
        self.cvae_list = []
        self.cvae_optimizer_list = []
        self.cvae_num = dataloader.load_vae_num()
        self.cvae_device_list = self.get_device_list(gpu_device, vae_num=self.cvae_num)
        print('device list', self.cvae_device_list, gpu_device)
        pbar = tqdm(total=len(self.cvae_dim_list), ascii=True)
        pbar.set_description('initiating cvaes...')
        for i, size in enumerate(self.cvae_dim_list):
            self.cvae_list.append(CVAE(size, latent_size, dataloader.load_cvae_condition_length(), i))
            if gpu:
                self.cvae_list[i].cuda(device=self.cvae_device_list[i])
            self.cvae_optimizer_list.append(optim.Adam(self.cvae_list[i].parameters(), lr=learning_rate))
            pbar.update()
        pbar.close()

    def get_device_list(self, gpu_devices, vae_num):
        ret = []
        li = gpu_devices.strip().split(',')
        for i in li:
            ret.append(int(i))
        return np.tile(np.array(ret), vae_num)[:vae_num]

    def train_single_vae_one_epoch(self, cvae_no, batch_size, gpu=False) -> np.ndarray:
        num_iter = self.train_set_size // batch_size
        for i in range(num_iter):
            batch_input = self.train_input[cvae_no][i * batch_size:(i + 1) * batch_size]
            batch_condition = self.train_condition[cvae_no][i * batch_size:(i + 1) * batch_size]
            if gpu:
                batch_input = batch_input.cuda(self.cvae_device_list[cvae_no])
                batch_condition = batch_condition.cuda(self.cvae_device_list[cvae_no])
            recon, mu, log_std = self.cvae_list[cvae_no](batch_input, batch_condition)
            self.cvae_optimizer_list[cvae_no].zero_grad()
            loss = self.cvae_list[cvae_no].loss_function(recon, batch_input, mu, log_std)
            loss.backward()
            self.cvae_optimizer_list[cvae_no].step()
            mse = torch.mean((recon - batch_input) ** 2).item()
            self.subpbar.set_postfix_str("mse loss: %.5e" % (mse))
            # print(self.pbar.postfix)
            self.subpbar.update()

        if num_iter * batch_size < self.train_set_size:
            batch_input = self.train_input[cvae_no][num_iter * batch_size:]
            batch_condition = self.train_condition[cvae_no][num_iter * batch_size:]
            # training_set_.extend(np.squeeze(batch_x[:, 0]).tolist())
            if gpu:
                batch_input = batch_input.cuda(self.cvae_device_list[cvae_no])
                batch_condition = batch_condition.cuda(self.cvae_device_list[cvae_no])
            recon, mu, log_std = self.cvae_list[cvae_no](batch_input, batch_condition)
            self.cvae_optimizer_list[cvae_no].zero_grad()
            loss = self.cvae_list[cvae_no].loss_function(recon, batch_input, mu, log_std)
            loss.backward()
            self.cvae_optimizer_list[cvae_no].step()

        if gpu:
            validate_input = self.validate_input[cvae_no].cuda(self.cvae_device_list[cvae_no])
            validate_condition = self.validate_condition[cvae_no].cuda(self.cvae_device_list[cvae_no])
        else:
            validate_input = self.validate_input[cvae_no]
            validate_condition = self.validate_condition[cvae_no]
        recon, mu, log_std = self.cvae_list[cvae_no](validate_input, validate_condition)
        return torch.squeeze(recon[:, 0]).cpu().detach().numpy()

    def train_vaes_in_serial(self, total_epoch, batch_size, gpu):
        self.start_time = time.time()
        with tqdm(total=total_epoch, ascii=True) as self.pbar:
            self.pbar.set_postfix_str("mse loss: -.-----e---")
            self.pbar.set_description("training cvaes")
            val_list = []
            for i in range(self.cvae_num):
                val_list.append(np.squeeze(self.validate_input[i][:, 0].numpy()))
            val_list = np.array(val_list)
            for epoch in range(total_epoch):
                # shuffle data every 5 epochs
                if (epoch + 1) % 5 == 0:
                    shuffle = np.random.permutation(self.train_set_size)
                    for j in range(self.cvae_num):
                        self.train_input[j] = self.train_input[j][shuffle]
                        self.train_condition[j] = self.train_condition[j][shuffle]
                recon_list = []
                num_iter = self.train_set_size // batch_size
                with tqdm(total=num_iter * self.cvae_num, ascii=True, leave=False) as self.subpbar:
                    for j in range(self.cvae_num):
                        recon_list.append(self.train_single_vae_one_epoch(j, batch_size, gpu))
                recon_list = np.array(recon_list)
                mse = np.mean((recon_list - val_list) ** 2).item()
                self.pbar.set_postfix_str("mse loss: %.5e" % (mse))
                self.pbar.update()

    def infer_in_serial(self, batch_size, gpu):
        self.recon = np.zeros(self.cvae_num).reshape((1, self.cvae_num))
        iters = self.test_set_size // batch_size
        with tqdm(total=iters, ascii=True) as pbar:
            pbar.set_description('icvae inferring')
            for i in range(iters):
                cache_list = []
                for cvae_no in range(self.cvae_num):
                    batch_input = self.test_input[cvae_no][i * batch_size:(i + 1) * batch_size]
                    batch_condition = self.test_condition[cvae_no][i * batch_size:(i + 1) * batch_size]
                    if gpu:
                        batch_input = batch_input.cuda(device=self.cvae_device_list[cvae_no])
                        batch_condition = batch_condition.cuda(device=self.cvae_device_list[cvae_no])
                    recon = self.cvae_list[cvae_no](batch_input, batch_condition)[0].cpu().detach().numpy()[:,
                            0].reshape((-1,))
                    cache_list.append(recon)
                self.recon = np.concatenate((self.recon, np.array(cache_list).transpose()), axis=0)
                pbar.update()
        if iters * batch_size < self.test_set_size:
            cache_list = []
            for cvae_no in range(self.cvae_num):
                batch_input = self.test_input[cvae_no][iters * batch_size:]
                batch_condition = self.test_condition[cvae_no][iters * batch_size:]
                if gpu:
                    batch_input = batch_input.cuda(device=self.cvae_device_list[cvae_no])
                    batch_condition = batch_condition.cuda(device=self.cvae_device_list[cvae_no])
                recon = self.cvae_list[cvae_no](batch_input, batch_condition)[0].cpu().detach().numpy()[:, 0].reshape(
                    (-1,))
                cache_list.append(recon)
            self.recon = np.concatenate((self.recon, np.array(cache_list).transpose()), axis=0)
        return self.recon[1:]

    def infer_in_serial_train_set(self, batch_size, gpu):
        self.recon_train = np.zeros(self.cvae_num).reshape((1, self.cvae_num))
        train_input, train_condition = self.dataloader.load_cvae_train_data()
        for i in range(len(train_input)):
            train_input[i] = (torch.squeeze(torch.Tensor(train_input[i])))
            train_condition[i] = (torch.squeeze(torch.Tensor(train_condition[i])))
        iters = self.test_set_size // batch_size
        with tqdm(total=iters, ascii=True) as pbar:
            pbar.set_description('icvae train set inferring')
            for i in range(iters):
                cache_list = []
                for cvae_no in range(self.cvae_num):
                    batch_input = train_input[cvae_no][i * batch_size:(i + 1) * batch_size]
                    batch_condition = train_condition[cvae_no][i * batch_size:(i + 1) * batch_size]
                    if gpu:
                        batch_input = batch_input.cuda(device=self.cvae_device_list[cvae_no])
                        batch_condition = batch_condition.cuda(device=self.cvae_device_list[cvae_no])
                    recon = self.cvae_list[cvae_no](batch_input, batch_condition)[0].cpu().detach().numpy()[:,
                            0].reshape((-1,))
                    cache_list.append(recon)
                self.recon_train = np.concatenate((self.recon_train, np.array(cache_list).transpose()), axis=0)
                pbar.update()
        if iters * batch_size < self.test_set_size:
            cache_list = []
            for cvae_no in range(self.cvae_num):
                batch_input = train_input[cvae_no][iters * batch_size:]
                batch_condition = train_condition[cvae_no][iters * batch_size:]
                if gpu:
                    batch_input = batch_input.cuda(device=self.cvae_device_list[cvae_no])
                    batch_condition = batch_condition.cuda(device=self.cvae_device_list[cvae_no])
                recon = self.cvae_list[cvae_no](batch_input, batch_condition)[0].cpu().detach().numpy()[:, 0].reshape(
                    (-1,))
                cache_list.append(recon)
            self.recon_train = np.concatenate((self.recon_train, np.array(cache_list).transpose()), axis=0)
        return self.recon_train[1:]


if __name__ == '__main__':
    vae_window_size = 20
    cnn_window_size = 30
    which_set = '1-1'
    normalize = True

    data_dir = '/remote-home/liuwenbo/pycproj/tsdata/data'
    dataset = 'smd'
    map_dir = '/remote-home/liuwenbo/pycproj/tsdata/maps/npmap'
    map = 'machine-' + which_set + '.npmap.pkl'
    train_set_file = os.path.join(data_dir, dataset, 'train/machine-' + which_set + '.pkl')
    test_set_file = os.path.join(data_dir, dataset, 'test/machine-' + which_set + '.pkl')
    label_file = os.path.join(data_dir, dataset, 'label/machine-' + which_set + '.pkl')
    map_file = os.path.join(map_dir, map)

    dataloader = DataLoader(train_set_file, test_set_file, label_file, normalize=normalize)
    dataloader.prepare_data(map_file, cnn_window_size=cnn_window_size, vae_window_size=vae_window_size)
    icvae = ICVAE(dataloader, 5, True, .01, '0,7')
