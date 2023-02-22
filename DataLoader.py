import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np


class DataLoader:
    def __init__(self, train_set_file, test_set_file, label_file, normalize=False):
        self.cnn_train_set = None
        f = open(train_set_file, 'rb')
        self.train_set: np.ndarray
        self.train_data = pickle.load(f).transpose()
        f.close()

        f = open(test_set_file, 'rb')
        self.test_set: np.ndarray
        self.test_data = pickle.load(f).transpose()
        f.close()

        f = open(label_file, 'rb')
        self.labels: np.ndarray
        self.labels = pickle.load(f)
        f.close()

        self.train_data_size = self.train_data.shape[1]
        self.test_data_size = self.test_data.shape[1]
        # print('size: ',self.train_data_size,self.test_data_size)

        # print(self.train_data.shape, self.test_data.shape, self.labels.shape)
        train_non_constant_var = []
        test_non_constant_var = []
        for i in self.train_data:
            train_non_constant_var.append(np.unique(i).shape[0])
        for i in self.test_data:
            test_non_constant_var.append(np.unique(i).shape[0])
        self.train_constant_var = np.where(np.array(train_non_constant_var) == 1)
        self.test_constant_var = np.where(np.array(test_non_constant_var) == 1)

        self.train_non_constant_var = np.setdiff1d(np.arange(len(train_non_constant_var)), self.train_constant_var)
        self.test_non_constant_var = np.setdiff1d(np.arange(len(train_non_constant_var)), self.test_constant_var)

        self.nc_train_data = self.train_data[self.train_non_constant_var]
        self.nc_test_data = self.test_data[self.train_non_constant_var]

        self.train_data_std = np.std(self.nc_train_data, axis=1).reshape(-1, 1)
        self.train_data_mean = np.mean(self.nc_train_data, axis=1).reshape(-1, 1)
        self.test_data_std = np.std(self.nc_train_data, axis=1).reshape(-1, 1)
        self.test_data_mean = np.mean(self.nc_train_data, axis=1).reshape(-1, 1)
        # both should use train set std and mean to normalize. because test set std and mean are abnormal
        print('\033[0;33mtrain/test set size\033[0m', self.nc_train_data.shape, self.nc_test_data.shape)
        if normalize:
            print('normalized')
            self.nc_train_data = (self.nc_train_data - self.train_data_mean) / self.train_data_std
            self.nc_test_data = (self.nc_test_data - self.test_data_mean) / self.test_data_std
        # print(np.squeeze(train_data_mean), np.squeeze(train_data_std), np.squeeze(test_data_mean),
        #       np.squeeze(test_data_std), sep='\n')

    def prepare_data(self, graph_file, vae_window_size=1, cnn_window_size=1):
        f = open(graph_file, 'rb')
        graph = pickle.load(f)
        f.close()
        # print(graph)
        parent_list = self.get_parents(graph)
        print('parent list len', len(parent_list))
        self.cvae_window_size = cnn_window_size - 1
        self.__vae_dim_list = []
        self.root_var = []
        self.non_root_var = []
        self.vae_train_set = []
        self.vae_test_set = []
        for index, list in enumerate(parent_list):
            if len(list) == 0:
                self.root_var.append(index)
            else:
                # print([index] + list)
                self.vae_train_set.append(self.nc_train_data[[index] + list].transpose())
                self.vae_test_set.append(self.nc_test_data[[index] + list].transpose())
                self.__vae_dim_list.append(len(list))
                self.non_root_var.append(index)
        # print(self.root_var)
        self.root_var = np.array(self.root_var)
        self.non_root_var = np.array(self.non_root_var)
        self.cnn_train_set = self.nc_train_data[self.root_var].transpose()
        self.cnn_test_set = self.nc_test_data[self.root_var].transpose()

        # prepare train dataset
        train_vae_window = np.arange(vae_window_size)
        train_vae_full_window = np.tile(train_vae_window, self.train_data_size - vae_window_size + 1).reshape(-1,
                                                                                                              vae_window_size)
        train_vae_add = np.repeat(np.arange(self.train_data_size - vae_window_size + 1), vae_window_size).reshape(-1,
                                                                                                                  vae_window_size)
        train_vae_index = train_vae_full_window + train_vae_add
        for i in range(len(self.vae_train_set)):
            self.vae_train_set[i] = self.vae_train_set[i][train_vae_index][cnn_window_size - vae_window_size + 1:]

        train_cnn_window = np.arange(cnn_window_size)
        train_cnn_full_window = np.tile(train_cnn_window, self.train_data_size - cnn_window_size + 1).reshape(-1,
                                                                                                              cnn_window_size)
        train_cnn_add = np.repeat(np.arange(self.train_data_size - cnn_window_size + 1), cnn_window_size).reshape(-1,
                                                                                                                  cnn_window_size)
        train_cnn_index = train_cnn_full_window + train_cnn_add
        self.cnn_train_set_y = self.cnn_train_set[cnn_window_size:]
        self.cnn_train_set_x = self.cnn_train_set[train_cnn_index[:-1]]

        # prepare test dataset
        test_vae_window = np.arange(vae_window_size)
        test_vae_full_window = np.tile(test_vae_window, self.test_data_size - vae_window_size + 1).reshape(-1,
                                                                                                           vae_window_size)
        test_vae_add = np.repeat(np.arange(self.test_data_size - vae_window_size + 1), vae_window_size).reshape(-1,
                                                                                                                vae_window_size)
        test_vae_index = test_vae_full_window + test_vae_add
        for i in range(len(self.vae_test_set)):
            self.vae_test_set[i] = self.vae_test_set[i][test_vae_index][cnn_window_size - vae_window_size + 1:]

        test_cnn_window = np.arange(cnn_window_size)
        test_cnn_full_window = np.tile(test_cnn_window, self.test_data_size - cnn_window_size + 1).reshape(-1,
                                                                                                           cnn_window_size)
        test_cnn_add = np.repeat(np.arange(self.test_data_size - cnn_window_size + 1), cnn_window_size).reshape(-1,
                                                                                                                cnn_window_size)
        test_cnn_index = test_cnn_full_window + test_cnn_add
        self.cnn_test_set_y = self.cnn_test_set[cnn_window_size:]
        self.cnn_test_set_x = self.cnn_test_set[test_cnn_index[:-1]]

        # label
        self.label_set = self.labels[cnn_window_size:]
        # print(self.labels.shape)

        self.train_set_size = self.cnn_train_set.shape[0]
        self.test_set_size = self.cnn_test_set.shape[0]

    def load_non_constant_train_data(self):
        return self.nc_train_data

    def load_max_and_min(self):
        return np.max([np.max(self.nc_train_data), np.max(self.nc_test_data)]), np.min(
            [np.min(self.nc_train_data), np.min(self.nc_test_data)])

    def load_train_set_norm_params(self):
        re_index = np.concatenate((self.root_var, self.non_root_var))
        return self.train_data_std[re_index], self.train_data_mean[re_index]

    def load_test_set_norm_params(self):
        re_index = np.concatenate((self.root_var, self.non_root_var))
        return self.test_data_std[re_index], self.test_data_mean[re_index]

    def load_vae_train_set(self):
        # print('\033[0;33mvae train set mem\033[0m',id(self.vae_train_set))
        return self.vae_train_set.copy()

    def load_train_set_ground_truth(self):
        train_set_ground_truth = []
        for i in self.vae_train_set:
            train_set_ground_truth.append(np.squeeze(i)[:, 0])
        return np.concatenate((self.cnn_train_set_y.transpose(), np.array(train_set_ground_truth)))

    def load_vae_test_set(self):
        return self.vae_test_set.copy()

    def load_cnn_train_set(self):
        return self.cnn_train_set_x.transpose(0, 2, 1), self.cnn_train_set_y

    def load_cnn_test_set(self):
        return self.cnn_test_set_x.transpose(0, 2, 1), self.cnn_test_set_y

    def load_label_set(self):
        return self.label_set

    def load_cnn_channel(self):
        # print(self.nc_train_data.shape)
        return self.nc_train_data.shape[0] - len(self.vae_train_set)

    def load_vae_num(self):
        return len(self.vae_test_set)

    def load_vae_dim_list(self):
        return np.array(self.__vae_dim_list) + 1

    def load_train_set_size(self):
        return self.vae_train_set[0].shape[0]

    def load_test_set_size(self):
        return self.vae_test_set[0].shape[0]

    def load_cnn_test_set_ground_truth(self):
        return self.cnn_test_set_y

    def load_vae_test_set_ground_truth(self):
        gt_list = []
        for i in range(self.load_vae_num()):
            gt_list.append(np.squeeze(self.vae_test_set[i][:, :, 0]))
            # print(self.vae_test_set[i][:,:,0].shape)
        return np.array(gt_list).transpose()

    def load_cvae_train_data(self):
        cvae_train_input = self.vae_train_set[:, -1]
        cvae_train_condition = self.vae_train_set[:, :-1]
        return cvae_train_input, cvae_train_condition

    def load_cvae_test_data(self):
        cvae_test_input = self.vae_test_set[:, -1]
        cvae_test_condition = self.vae_test_set[:, :-1]
        return cvae_test_input, cvae_test_condition

    def load_cvae_test_ground_truth(self):
        return self.vae_test_set[:, -1]

    def load_obvious_anomaly_positions(self):
        anomaly_vars = np.setdiff1d(self.test_non_constant_var, self.train_non_constant_var)
        anomaly_position_list = []
        for i in anomaly_vars:
            normal_value = self.train_data[i][0]
            # print('test data shape',self.test_data.shape)
            # print(np.where(self.test_data[i]!=normal_value))
            anomaly_position_list += (np.where(self.test_data[i] != normal_value)[0].tolist())
        return np.unique(anomaly_position_list).astype(int)

    def get_parents(self, graph):
        nodes = graph.shape[0]
        parents_list = []
        for i in range(nodes):
            parents_list.append([])
        for i in range(nodes):
            for j in range(nodes):
                if graph[i][j] == -1:
                    parents_list[j].append(i)
        return parents_list

    def draw_train_set(self):
        for i in range(self.load_vae_num()):
            fig, ax = plt.subplots()
            print(self.vae_train_set[i].shape)
            y = self.vae_train_set[i][:, :, 0]
            x = np.arange(self.load_train_set_size())
            ax.plot(x, y, linewidth=1)
            fig.set_figwidth(10)
            fig.set_figheight(5)
            fig.savefig('save/trainset_' + str(i + 2) + '.png', dpi=600)
            plt.close(fig)


if __name__ == '__main__':
    which_set = '1-1'

    data_dir = '/remote-home/liuwenbo/pycproj/tsdata/data'
    dataset = 'smd'
    map_dir = '/remote-home/liuwenbo/pycproj/tsdata/maps/npmap'
    map = 'machine-' + which_set + '.npmap.pkl'
    train_set_file = os.path.join(data_dir, dataset, 'train/machine-' + which_set + '.pkl')
    test_set_file = os.path.join(data_dir, dataset, 'test/machine-' + which_set + '.pkl')
    label_file = os.path.join(data_dir, dataset, 'label/machine-' + which_set + '.pkl')
    map_file = os.path.join(map_dir, map)

    dataloader = DataLoader(train_set_file, test_set_file, label_file, normalize=False)
    dataloader.prepare_data(map_file, cnn_window_size=20, vae_window_size=1)
    # train_std,train_mean=dataloader.load_train_set_norm_params()
    # test_std,test_mean=dataloader.load_test_set_norm_params()
    # print(train_std.shape,train_mean.shape)
    # print(test_std.shape,test_mean.shape)
    cnn_train_x, cnn_train_y = dataloader.load_cnn_train_set()
    print(cnn_train_x.shape, cnn_train_y.shape)
    print(dataloader.load_train_set_ground_truth().shape)
    nc_train_data = dataloader.load_non_constant_train_data().transpose()
    print(np.max(nc_train_data, axis=0))
    print(np.min(nc_train_data, axis=0))
    print(dataloader.load_max_and_min())
