import os
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import torch.multiprocessing as mp
import torch.nn.functional as F

from VAE import VAE
from Draw import DrawTrainMSELoss


class IVAE:
    def __init__(self, dataloader, latent_size, gpu, learning_rate, gpu_device):
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.train_set_size = dataloader.load_train_set_size()
        self.test_set_size = dataloader.load_test_set_size()
        # load data
        vae_train_set = dataloader.load_vae_train_set()
        vae_test_set = dataloader.load_vae_test_set()
        label_set = dataloader.load_label_set()

        self.vae_dim_list = dataloader.load_vae_dim_list()
        print('vae dim list:', self.vae_dim_list)

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
        self.vae_optimizer_list = []
        self.vae_num = dataloader.load_vae_num()
        self.vae_device_list = self.get_device_list(gpu_device, vae_num=self.vae_num)
        print('device list', self.vae_device_list, gpu_device)
        for i, size in enumerate(self.vae_dim_list):
            self.vae_list.append(VAE(size, latent_size, i))
            if gpu:
                self.vae_list[i].cuda(device=self.vae_device_list[i])
            self.vae_optimizer_list.append(optim.Adam(self.vae_list[i].parameters(), lr=learning_rate))

    def get_device_list(self, gpu_devices, vae_num):
        ret = []
        li = gpu_devices.strip().split(',')
        for i in li:
            ret.append(int(i))
        return np.tile(np.array(ret), vae_num)[:vae_num]

    def train_single_vae_one_epoch(self, vae_no, batch_size, gpu=False, pbar=False) -> np.ndarray:
        num_iter = self.train_set_size // batch_size
        if pbar:
            for i in range(num_iter):
                batch_x = self.vae_train_set[vae_no][i * batch_size:(i + 1) * batch_size]
                # training_set_.extend(np.squeeze(batch_x[:, 0]).tolist())
                if gpu:
                    batch_x = batch_x.cuda(self.vae_device_list[vae_no])
                recon, mu, log_std = self.vae_list[vae_no](batch_x)
                self.vae_optimizer_list[vae_no].zero_grad()
                loss = self.vae_list[vae_no].loss_function(recon, batch_x, mu, log_std)
                loss.backward()
                self.vae_optimizer_list[vae_no].step()
                mse = torch.mean((recon - batch_x) ** 2).item()
                self.subpbar.set_postfix_str("mse loss: %.5e" % (mse))
                # print(self.pbar.postfix)
                self.subpbar.update()

            if num_iter * batch_size < self.train_set_size:
                batch_x = self.vae_train_set[vae_no][num_iter * batch_size:]
                # training_set_.extend(np.squeeze(batch_x[:, 0]).tolist())
                if gpu:
                    batch_x = batch_x.cuda(self.vae_device_list[vae_no])
                recon, mu, log_std = self.vae_list[vae_no](batch_x)
                self.vae_optimizer_list[vae_no].zero_grad()
                loss = self.vae_list[vae_no].loss_function(recon, batch_x, mu, log_std)
                loss.backward()
                self.vae_optimizer_list[vae_no].step()
            # training_set_=np.array(training_set_)
            '''fig,ax=plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(5)
            x=np.arange(training_set_.shape[0])
            ax.plot(x,training_set_,linewidth=1)
            fig.savefig(save_file_name,dpi=600)
            plt.close(fig)'''
        else:
            for i in range(num_iter):
                batch_x = self.vae_train_set[vae_no][i * batch_size:(i + 1) * batch_size]
                if gpu:
                    batch_x = batch_x.cuda(self.vae_device_list[vae_no])
                recon, mu, log_std = self.vae_list[vae_no](batch_x)
                self.vae_optimizer_list[vae_no].zero_grad()
                loss = self.vae_list[vae_no].loss_function(recon, batch_x, mu, log_std)
                loss.backward()
                self.vae_optimizer_list[vae_no].step()

            if num_iter * batch_size < self.train_set_size:
                batch_x = self.vae_train_set[vae_no][num_iter * batch_size:]
                if gpu:
                    batch_x = batch_x.cuda(self.vae_device_list[vae_no])
                recon, mu, log_std = self.vae_list[vae_no](batch_x)
                self.vae_optimizer_list[vae_no].zero_grad()
                loss = self.vae_list[vae_no].loss_function(recon, batch_x, mu, log_std)
                loss.backward()
                self.vae_optimizer_list[vae_no].step()

        if gpu:
            validate_set = self.vae_validate_set[vae_no].cuda(self.vae_device_list[vae_no])
        else:
            validate_set = self.vae_validate_set[vae_no]
        recon, mu, log_std = self.vae_list[vae_no](validate_set)
        return torch.squeeze(recon[:, 0]).cpu().detach().numpy()

    # test train
    def train_vaes_in_parallel(self, epoch, batch_size, gpu, proc=None):
        for i in range(self.vae_num):
            self.vae_list[i].share_memory()
        self.pbar = tqdm(total=epoch, ascii=True)
        self.pbar.set_postfix_str("mse loss: -.-----e---")
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
                for j in range(self.vae_num):
                    self.vae_train_set[j] = self.vae_train_set[j][shuffle]

            pool = mp.Pool(proc)
            arg_list = []
            # get recon list and train in parallel
            for j in range(self.vae_num):
                arg_list.append((j, batch_size, gpu, False))
            # print('arglist:', arg_list)
            recon_list = np.array(pool.map(self.train_single_vae_one_epoch, arg_list))
            pool.close()

            # result_list=[]
            # recon_list=[]
            # for j in range(self.vae_num):
            #     result_list.append(pool.apply_async(self.train_single_vae_one_epoch,(j, batch_size, gpu, False)))
            # for j in range(self.vae_num):
            #     recon_list.append(result_list[j].get())

            recon_list = np.array(recon_list)
            print(recon_list.shape)
            mse = np.mean((recon_list - val_list) ** 2)
            self.pbar.set_postfix_str("mse loss: %.5e" % (mse))
            self.pbar.update()
        self.pbar.close()

    def train_vaes_in_serial(self, total_epoch, batch_size, gpu, figfile):
        self.start_time = time.time()
        with tqdm(total=total_epoch, ascii=True) as self.pbar:
            self.pbar.set_postfix_str("mse loss: -.-----e---")
            self.pbar.set_description("training vaes")
            val_list = []
            for i in range(self.vae_num):
                val_list.append(np.squeeze(self.vae_validate_set[i][:, 0].numpy()))
            val_list = np.array(val_list)
            draw = DrawTrainMSELoss(np.mean(val_list ** 2))
            # print('val list shape', val_list.shape)
            for epoch in range(total_epoch):
                # shuffle data every 5 epochs
                if (epoch + 1) % 5 == 0:
                    shuffle = np.random.permutation(self.train_set_size)
                    for j in range(self.vae_num):
                        self.vae_train_set[j] = self.vae_train_set[j][shuffle]
                recon_list = []
                num_iter = self.train_set_size // batch_size
                with tqdm(total=num_iter * self.vae_num, ascii=True, leave=False) as self.subpbar:
                    for j in range(self.vae_num):
                        recon_list.append(self.train_single_vae_one_epoch(j, batch_size, gpu, True))
                recon_list = np.array(recon_list)
                mse = np.mean((recon_list - val_list) ** 2).item()
                draw.add_mse_loss(mse)
                self.pbar.update()
        draw.draw(figfile)

    def infer_in_serial(self, batch_size, gpu):
        self.recon = np.zeros(self.vae_num).reshape((1, self.vae_num))
        iters = self.test_set_size // batch_size
        with tqdm(total=iters, ascii=True) as pbar:
            pbar.set_description('ivae inferring')
            for i in range(iters):
                cache_list = []
                for vae_no in range(self.vae_num):
                    batch_x = self.vae_test_set[vae_no][i * batch_size:(i + 1) * batch_size]
                    if gpu:
                        batch_x = batch_x.cuda(device=self.vae_device_list[vae_no])
                    recon = self.vae_list[vae_no](batch_x)[0].cpu().detach().numpy()[:, 0].reshape((-1,))
                    cache_list.append(recon)
                self.recon = np.concatenate((self.recon, np.array(cache_list).transpose()), axis=0)
                pbar.update()
        if iters * batch_size < self.test_set_size:
            cache_list = []
            for vae_no in range(self.vae_num):
                batch_x = self.vae_test_set[vae_no][iters * batch_size:]
                if gpu:
                    batch_x = batch_x.cuda(self.vae_device_list[vae_no])
                recon = self.vae_list[vae_no](batch_x)[0].cpu().detach().numpy()[:, 0].reshape((-1,))
                cache_list.append(recon)
            self.recon = np.concatenate((self.recon, np.array(cache_list).transpose()), axis=0)
        return self.recon[1:]
