import multiprocessing as mp
import os
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from DataLoader import DataLoader
from VAE import VAE


class IVAE:
    def __init__(self, train_set_file, test_set_file, label_file, map_file, latent_size, gpu, learning_rate,
                 window_size, gpu_device, normalize):
        print(normalize)
        dataloader = DataLoader(train_set_file, test_set_file, label_file, normalize=normalize)
        dataloader.prepare_data(map_file, cnn_window_size=window_size, vae_window_size=1)
        self.train_set_size = dataloader.load_train_set_size()
        # load data
        cnn_train_set_x, cnn_train_set_y = dataloader.load_cnn_train_set()
        cnn_test_set_x, cnn_test_set_y = dataloader.load_cnn_test_set()
        vae_train_set = dataloader.load_vae_train_set()
        vae_test_set = dataloader.load_vae_test_set()
        label_set = dataloader.load_label_set()

        self.vae_dim_list = dataloader.load_vae_dim_list()
        print('vae dim list:', self.vae_dim_list)

        self.cnn_train_set_x = torch.Tensor(cnn_train_set_x)
        self.cnn_train_set_y = torch.Tensor(cnn_train_set_y)
        self.cnn_test_set_x = torch.Tensor(cnn_test_set_x)
        self.cnn_test_set_y = torch.Tensor(cnn_test_set_y)
        self.label_set = torch.Tensor(label_set)
        self.vae_train_set = []
        self.vae_test_set = []
        self.vae_validate_set = []
        for i in range(len(vae_train_set)):
            self.vae_train_set.append(torch.squeeze(torch.Tensor(vae_train_set[i])))
            self.vae_test_set.append(torch.squeeze(torch.Tensor(vae_test_set[i])))
            self.vae_validate_set.append(self.vae_test_set[-1][:100])

        # create vaes
        self.vae_list = []
        self.optimizer_list = []
        self.device_list = self.get_device_list(gpu_device, vae_num=self.vae_dim_list.shape[0])
        self.vae_num = self.vae_dim_list.shape[0]
        print('device list', self.device_list, gpu_device)
        for i, size in enumerate(self.vae_dim_list):
            self.vae_list.append(VAE(size, latent_size, i))
            if gpu:
                self.vae_list[i].cuda(device=self.device_list[i])
            self.optimizer_list.append(optim.Adam(self.vae_list[i].parameters(), lr=learning_rate))

    def train_single_vae_one_epoch(self, vae_no, batch_size, gpu=False, pbar=False) -> np.ndarray:
        num_iter = self.train_set_size // batch_size
        if pbar:
            for i in range(num_iter):
                batch_x = self.vae_train_set[vae_no][i * batch_size:(i + 1) * batch_size]
                if gpu:
                    batch_x = batch_x.cuda(self.device_list[vae_no])
                recon, mu, log_std = self.vae_list[vae_no](batch_x)
                self.optimizer_list[vae_no].zero_grad()
                loss = self.vae_list[vae_no].loss_function(recon, batch_x, mu, log_std)
                loss.backward()
                self.optimizer_list[vae_no].step()
                mse=torch.mean((recon-batch_x)**2).item()
                self.subpbar.set_postfix_str("mse loss: %.5e" % (mse))
                self.subpbar.update()

            if num_iter * batch_size < self.train_set_size:
                batch_x = self.vae_train_set[vae_no][num_iter * batch_size:]
                if gpu:
                    batch_x = batch_x.cuda(self.device_list[vae_no])
                recon, mu, log_std = self.vae_list[vae_no](batch_x)
                self.optimizer_list[vae_no].zero_grad()
                loss = self.vae_list[vae_no].loss_function(recon, batch_x, mu, log_std)
                loss.backward()
                self.optimizer_list[vae_no].step()
        else:
            for i in range(num_iter):
                batch_x = self.vae_train_set[vae_no][i * batch_size:(i + 1) * batch_size]
                if gpu:
                    batch_x = batch_x.cuda(self.device_list[vae_no])
                recon, mu, log_std = self.vae_list[vae_no](batch_x)
                self.optimizer_list[vae_no].zero_grad()
                loss = self.vae_list[vae_no].loss_function(recon, batch_x, mu, log_std)
                loss.backward()
                self.optimizer_list[vae_no].step()

            if num_iter * batch_size < self.train_set_size:
                batch_x = self.vae_train_set[vae_no][num_iter * batch_size:]
                if gpu:
                    batch_x = batch_x.cuda(self.device_list[vae_no])
                recon, mu, log_std = self.vae_list[vae_no](batch_x)
                self.optimizer_list[vae_no].zero_grad()
                loss = self.vae_list[vae_no].loss_function(recon, batch_x, mu, log_std)
                loss.backward()
                self.optimizer_list[vae_no].step()

        if gpu:
            validate_set = self.vae_validate_set[vae_no].cuda(self.device_list[vae_no])
        else:
            validate_set = self.vae_validate_set[vae_no].cuda(self.device_list[vae_no])
        recon, mu, log_std = self.vae_list[vae_no](validate_set)
        # print(recon.shape)
        # print('recon:',recon[1].cpu().detach().numpy(),'\nval:',validate_set[1].cpu().numpy())
        return torch.squeeze(recon[:, 0]).cpu().detach().numpy()

    # test train
    def train_in_parallel(self, epoch, batch_size, gpu, proc=None):
        pbar = tqdm(total=epoch, ascii=True)
        pbar.set_postfix_str("mse loss: -.-----e---")
        # build validate set, useful data only
        val_list = []
        for i in range(self.vae_num):
            val_list.append(np.squeeze(self.vae_validate_set[i][:, 0].numpy()))
        val_list = np.array(val_list)
        print(val_list.shape)
        for i in range(epoch):
            # shuffle data every 5 epochs
            if i % 5 == 0:
                shuffle = np.random.permutation(self.train_set_size)
                for i in range(self.vae_num):
                    self.vae_train_set[i] = self.vae_train_set[i][shuffle]
            pool = mp.Pool(proc)
            arg_list = []
            for i in range(self.vae_num):
                arg_list.append((i, batch_size, gpu))
            recon_list = np.array(pool.map(self.train_single_vae_one_epoch, arg_list))
            print(recon_list.shape)
            mse = 1.
            # train single vae
            recon_list = np.array(recon_list)
            mse = np.mean((recon_list - val_list) ** 2)
            pbar.set_postfix_str("mse loss: %.5e" % (mse))
            pbar.update()
        pbar.close()

    def train_in_serial(self, epoch, batch_size, gpu):
        with tqdm(total=epoch, ascii=True) as pbar:
            pbar.set_postfix_str("mse loss: -.-----e---")
            val_list = []
            for i in range(self.vae_num):
                val_list.append(np.squeeze(self.vae_validate_set[i][:, 0].numpy()))
            val_list = np.array(val_list)
            # print('val list shape', val_list.shape)
            for i in range(epoch):
                # shuffle data every 5 epochs
                if i % 5 == 0:
                    shuffle = np.random.permutation(self.train_set_size)
                    for i in range(self.vae_num):
                        self.vae_train_set[i] = self.vae_train_set[i][shuffle]
                recon_list = []
                num_iter = self.train_set_size // batch_size
                with tqdm(total=num_iter * self.vae_num, ascii=True, leave=False) as self.subpbar:
                    for i in range(self.vae_num):
                        recon_list.append(self.train_single_vae_one_epoch(i, batch_size, gpu, True))
                recon_list = np.array(recon_list)
                mse = np.mean((recon_list - val_list) ** 2)
                pbar.set_postfix_str("mse loss: %.5e" % (mse))
                pbar.update()

    def get_device_list(self, gpu_devices, vae_num):
        ret = []
        li = gpu_devices.strip().split(',')
        for i in li:
            ret.append(int(i))
        return np.tile(np.array(ret), vae_num)[:vae_num]
