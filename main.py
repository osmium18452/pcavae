import multiprocessing as mp
import os
import pickle

import numpy as np

data_dir='/remote-home/liuwenbo/pycproj/tsdata/data'
dataset='smd'
train_set_file='train/machine-1-1.pkl'

if __name__ == '__main__':
    f=open(os.path.join(data_dir,dataset,train_set_file),'rb')
    train_set:np.ndarray
    train_set=pickle.load(f)
    f.close()
    print(train_set.shape)