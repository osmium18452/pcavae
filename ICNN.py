import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from CNN import CNN


class ICNN:
    def __init__(self, dataloader, window_size, gpu, learning_rate, gpu_device):
        self.recon = None
        self.dataloader=dataloader
        self.train_set_size = dataloader.load_train_set_size()
        cnn_train_set_x, cnn_train_set_y = dataloader.load_cnn_train_set()
        cnn_test_set_x, cnn_test_set_y = dataloader.load_cnn_test_set()
        self.cnn_train_set_x = torch.Tensor(cnn_train_set_x)
        self.cnn_train_set_y = torch.Tensor(cnn_train_set_y)
        self.cnn_test_set_x = torch.Tensor(cnn_test_set_x)
        self.cnn_test_set_y = torch.Tensor(cnn_test_set_y)
        self.cnn_validate_set_x = self.cnn_test_set_x[100:]
        self.cnn_validate_set_y = self.cnn_test_set_y[100:]
        self.cnn_channel = dataloader.load_cnn_channel()
        self.test_set_size=dataloader.load_test_set_size()

        print('cnn train set shape', self.cnn_train_set_x.shape, self.cnn_train_set_y.shape)

        self.cnn = CNN(self.cnn_channel, window_size)
        if (gpu):
            self.device = int(gpu_device.strip().split(',')[0])
            self.cnn.cuda(device=self.device)
            self.cnn_validate_set_x = self.cnn_test_set_x.cuda(device=self.device)
            self.cnn_validate_set_y = self.cnn_test_set_y.cuda(device=self.device)
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=learning_rate)

    def train(self, epoch, batch_size, gpu):
        with tqdm(total=epoch, ascii=True) as pbar:
            pbar.set_description('training cnn')
            pbar.set_postfix_str("mse loss: -.-----e---")
            for i in range(epoch):
                # print('batch size:',batch_size)
                if i % 5 == 0:
                    shuffle = np.random.permutation(self.train_set_size)
                    self.cnn_train_set_x = self.cnn_train_set_x[shuffle]
                    self.cnn_train_set_y = self.cnn_train_set_y[shuffle]
                num_iter = self.train_set_size // batch_size
                with tqdm(total=num_iter, ascii=True, leave=False) as subpbar:
                    for j in range(num_iter):
                        batch_x = self.cnn_train_set_x[j * batch_size:(j + 1) * batch_size]
                        batch_y = self.cnn_train_set_y[j * batch_size:(j + 1) * batch_size]
                        if gpu:
                            batch_x = batch_x.cuda(self.device)
                            batch_y = batch_y.cuda(self.device)
                        recon = self.cnn(batch_x)
                        self.optimizer.zero_grad()
                        loss = self.cnn.loss_function(recon, batch_y)
                        loss.backward()
                        self.optimizer.step()
                        subpbar.set_postfix_str('mse loss: %.5e' % loss.item())
                        subpbar.update()
                if num_iter * batch_size < self.train_set_size:
                    batch_x = self.cnn_train_set_x[num_iter * batch_size:]
                    batch_y = self.cnn_train_set_y[num_iter * batch_size:]
                    if gpu:
                        batch_x = batch_x.cuda(self.device)
                        batch_y = batch_y.cuda(self.device)
                    recon = self.cnn(batch_x)
                    self.optimizer.zero_grad()
                    loss = self.cnn.loss_function(recon, batch_y)
                    loss.backward()
                    self.optimizer.step()
                # print('cnn validate set x',self.cnn_validate_set_x.shape)
                recon_val = self.cnn(self.cnn_validate_set_x)
                mse = np.mean((recon_val.cpu().detach().numpy() - self.cnn_validate_set_y.cpu().detach().numpy()) ** 2)
                pbar.set_postfix_str('mse loss: %.5e' % mse)
                pbar.update()

    def infer(self,batch_size,gpu):
        self.recon=np.zeros(self.cnn_channel).reshape((1,self.cnn_channel))
        iters=self.test_set_size//batch_size
        with tqdm(total=iters,ascii=True) as pbar:
            pbar.set_description('cnn inferring')
            for i in range(iters):
                batch_x=self.cnn_test_set_x[i*batch_size:(i+1)*batch_size]
                if gpu:
                    batch_x=batch_x.cuda(self.device)
                recon=self.cnn(batch_x).cpu().detach().numpy()
                self.recon=np.concatenate((self.recon,recon),axis=0)
                pbar.update()
        if iters*batch_size<self.test_set_size:
            batch_x=self.cnn_test_set_x[iters*batch_size:]
            if gpu:
                batch_x=batch_x.cuda(self.device)
            recon=self.cnn(batch_x).cpu().detach().numpy()
            self.recon=np.concatenate((self.recon,recon),axis=0)
        self.recon=np.array(self.recon[1:])
        return self.recon

    def infer_train_set(self,batch_size,gpu):
        self.train_recon=np.zeros(self.cnn_channel).reshape((1,self.cnn_channel))
        iters=self.train_set_size//batch_size
        train_set_x,train_set_y=self.dataloader.load_cnn_train_set()
        train_set_x=torch.Tensor(train_set_x)
        with tqdm(total=iters,ascii=True) as pbar:
            pbar.set_description('cnn inferring')
            for i in range(iters):
                batch_x=train_set_x[i*batch_size:(i+1)*batch_size]
                if gpu:
                    batch_x=batch_x.cuda(self.device)
                recon=self.cnn(batch_x).cpu().detach().numpy()
                self.train_recon=np.concatenate((self.train_recon,recon),axis=0)
                pbar.update()
        if iters*batch_size<self.train_set_size:
            batch_x=train_set_x[iters*batch_size:]
            if gpu:
                batch_x=batch_x.cuda(self.device)
            recon=self.cnn(batch_x).cpu().detach().numpy()
            self.train_recon=np.concatenate((self.train_recon,recon),axis=0)
        self.train_recon=np.array(self.train_recon[1:])
        return self.train_recon




        # create cnns
        # self.cnn_list = []
        # self.cnn_optimizer_list = []
        # self.cnn_num = dataloader.load_cnn_num()
        # self.cnn_device_list = self.get_device_list(gpu_device, vae_num=self.cnn_num)
        # print('cnn device list:', self.cnn_device_list)
        # for i in range(self.cnn_num):
        #     self.cnn_list.append(CNN(window_size))
        #     if gpu:
        #         self.cnn_list[i].cuda(device=self.cnn_device_list[i])
        #     self.cnn_optimizer_list.append(optim.Adam(self.cnn_list[i].parameters(), lr=learning_rate))

    '''
    def train_single_cnn_one_epoch(self, cnn_no, batch_size, gpu, pbar):
        num_iter = self.train_set_size // batch_size
        if pbar:
            for i in range(num_iter):
                batch_x = self.cnn_train_set_x[cnn_no][i * batch_size:(i + 1) * batch_size]
                batch_y = self.cnn_train_set_y[cnn_no][i * batch_size:(i + 1) * batch_size]
                if gpu:
                    batch_x = batch_x.cuda(self.cnn_device_list[cnn_no])
                recon = self.cnn_list[cnn_no](batch_x)
                self.cnn_optimizer_list[cnn_no].zero_grad()
                loss = self.cnn_list[cnn_no].loss_function(recon, batch_y)
                loss.backward()
                self.cnn_optimizer_list[cnn_no].step()
                mse = torch.mean((recon - batch_x) ** 2).item()
                self.subpbar.set_postfix_str("mse loss: %.5e" % (mse))
                self.subpbar.update()

            if num_iter * batch_size < self.train_set_size:
                batch_x = self.cnn_train_set_x[cnn_no][num_iter * batch_size:]
                batch_y = self.cnn_train_set_y[cnn_no][num_iter * batch_size:]
                if gpu:
                    batch_x = batch_x.cuda(self.cnn_device_list[cnn_no])
                recon = self.cnn_list[cnn_no](batch_x)
                self.cnn_optimizer_list[cnn_no].zero_grad()
                loss = self.cnn_list[cnn_no].loss_function(recon, batch_y)
                loss.backward()
                self.cnn_optimizer_list[cnn_no].step()
        else:
            for i in range(num_iter):
                batch_x = self.cnn_train_set_x[cnn_no][i * batch_size:(i + 1) * batch_size]
                batch_y = self.cnn_train_set_y[cnn_no][i * batch_size:(i + 1) * batch_size]
                if gpu:
                    batch_x = batch_x.cuda(self.cnn_device_list[cnn_no])
                recon = self.cnn_list[cnn_no](batch_x)
                self.cnn_optimizer_list[cnn_no].zero_grad()
                loss = self.cnn_list[cnn_no].loss_function(recon, batch_y)
                loss.backward()
                self.cnn_optimizer_list[cnn_no].step()

            if num_iter * batch_size < self.train_set_size:
                batch_x = self.cnn_train_set_x[cnn_no][num_iter * batch_size:]
                batch_y = self.cnn_train_set_y[cnn_no][num_iter * batch_size:]
                if gpu:
                    batch_x = batch_x.cuda(self.cnn_device_list[cnn_no])
                recon = self.cnn_list[cnn_no](batch_x)
                self.cnn_optimizer_list[cnn_no].zero_grad()
                loss = self.cnn_list[cnn_no].loss_function(recon, batch_y)
                loss.backward()
                self.cnn_optimizer_list[cnn_no].step()

        if gpu:
            validate_set = self.cnn_validate_set_x[cnn_no].cuda(self.cnn_device_list[cnn_no])
        else:
            validate_set = self.cnn_validate_set_x[cnn_no]
        recon = self.cnn_list[cnn_no](validate_set)
        print('cnn recon shape', recon.shape)
        return torch.squeeze(recon).cpu().detach().numpy()
    '''
    '''
    def train_cnns_in_serial(self, epoch, batch_size, gpu):
        with tqdm(total=epoch, ascii=True) as pbar:
            pbar.set_postfix_str("mse loss: -.-----e---")
            pbar.set_description("training cnns")
            val_list = self.cnn_validate_set_y
            print('cnn val shape', val_list.shape)
            for i in range(epoch):
                # shuffle data every 5 epochs
                if i % 5 == 0:
                    shuffle = np.random.permutation(self.train_set_size)
                    self.cnn_train_set_x=self.cnn_train_set_x[:,shuffle]
                    # for i in range(self.cnn_num):
                    #     self.cnn_train_set_x[i] = self.cnn_train_set_x[i][shuffle]
                    #     self.cnn_train_set_y[i] =self.cnn_train_set_y[i][shuffle]
                recon_list = []
                num_iter = self.train_set_size // batch_size
                with tqdm(total=num_iter * self.cnn_num, ascii=True, leave=False) as self.subpbar:
                    for i in range(self.cnn_num):
                        recon_list.append(self.train_single_cnn_one_epoch(i, batch_size, gpu, True))
                recon_list = np.array(recon_list)
                mse = np.mean((recon_list - val_list) ** 2)
                pbar.set_postfix_str("mse loss: %.5e" % (mse))
                pbar.update()
    '''
