import os.path
import pickle

import numpy as np


class DataLoader:
    def __init__(self, train_set_file, test_set_file, label_file):
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
        self.labels = pickle.load(f).transpose()
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
        # print(type(self.train_non_constant_var))
        # print(self.train_constant_var)
        # print(type(self.test_non_constant_var))
        # print(self.test_constant_var)

        self.nc_train_data = self.train_data[self.train_non_constant_var]
        self.nc_test_data = self.test_data[self.train_non_constant_var]
        # print(self.nc_train_data.shape)
        # print(self.nc_test_data.shape)

    def prepare_data(self, graph_file, vae_window_size=1, cnn_window_size=1):
        f = open(graph_file, 'rb')
        graph = pickle.load(f)['G'].graph
        f.close()
        # print(graph)
        parent_list = self.get_parents(graph)
        self.__vae_dim_list=[]
        # for i in parent_list:
        #     self.vae_dim_list.append(len(i))
        # print(self.vae_dim_list)
        # print(parent_list)
        self.root_var = []
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
        # print(self.root_var)
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
            self.vae_train_set[i] = self.vae_train_set[i][train_vae_index][cnn_window_size - vae_window_size:]

        train_cnn_window = np.arange(cnn_window_size)
        train_cnn_full_window = np.tile(train_cnn_window, self.train_data_size - cnn_window_size + 1).reshape(-1,
                                                                                                              cnn_window_size)
        train_cnn_add = np.repeat(np.arange(self.train_data_size - cnn_window_size + 1), cnn_window_size).reshape(-1,
                                                                                                                  cnn_window_size)
        train_cnn_index = train_cnn_full_window + train_cnn_add

        self.cnn_train_set = self.cnn_train_set[train_cnn_index]

        # prepare test dataset
        test_vae_window = np.arange(vae_window_size)
        test_vae_full_window = np.tile(test_vae_window, self.test_data_size - vae_window_size + 1).reshape(-1,
                                                                                                           vae_window_size)
        test_vae_add = np.repeat(np.arange(self.test_data_size - vae_window_size + 1), vae_window_size).reshape(-1,
                                                                                                                vae_window_size)
        test_vae_index = test_vae_full_window + test_vae_add

        for i in range(len(self.vae_test_set)):
            self.vae_test_set[i] = self.vae_test_set[i][test_vae_index][cnn_window_size - vae_window_size:]

        test_cnn_window = np.arange(cnn_window_size)
        test_cnn_full_window = np.tile(test_cnn_window, self.test_data_size - cnn_window_size + 1).reshape(-1,
                                                                                                           cnn_window_size)
        test_cnn_add = np.repeat(np.arange(self.test_data_size - cnn_window_size + 1), cnn_window_size).reshape(-1,
                                                                                                                cnn_window_size)
        test_cnn_index = test_cnn_full_window + test_cnn_add

        self.cnn_test_set = self.cnn_test_set[test_cnn_index]

        self.label_set = self.labels[cnn_window_size:]

        self.train_set_size = self.cnn_train_set.shape[0]
        self.test_set_size = self.cnn_test_set.shape[0]

    def load_vae_train_set(self):
        return self.vae_train_set

    def load_vae_test_set(self):
        return self.vae_test_set

    def load_cnn_train_set(self):
        return self.cnn_train_set

    def load_cnn_test_set(self):
        return self.cnn_test_set

    def load_label_set(self):
        return self.label_set

    def load_vae_num(self):
        return len(self.vae_test_set)

    def load_vae_dim_list(self):
        return np.array(self.__vae_dim_list) + 1

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
    print(cnn_test_set.shape, cnn_train_set.shape,dataloader.load_vae_num())
    for i in range(dataloader.load_vae_num()):
        print(vae_train_set[i].shape,vae_test_set[i].shape)
    print(dataloader.load_vae_dim_list())
