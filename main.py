import argparse
import multiprocessing as mp
import os
import pickle
import matplotlib.pyplot as plt

import torch
from torch import optim

from DataLoader import DataLoader
from VAE import VAE
from IVAE import IVAE
from ICNN import ICNN

import numpy as np


def draw_gt_and_recon(gt, recon, labels=None, predicted_anomaly_positions=None, obvious_anomaly_positions=None,
                      save_file=None):
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    fig.set_dpi(300)
    x = np.arange(gt.shape[0])
    ax.plot(x, gt, label='ground truth', zorder=0,linewidth=1)
    ax.plot(x, recon, alpha=0.7, label='reconstruction', zorder=1,linewidth=1)
    if labels is not None:
        x_labels = np.where(labels == 1)
        y_labels = gt[x_labels]
        ax.scatter(x_labels, y_labels, color='red', label='true anomaly', zorder=100, s=.5)
    if predicted_anomaly_positions is not None:
        y = recon[predicted_anomaly_positions]
        ax.scatter(predicted_anomaly_positions, y, label='pred anomaly', color='darkgreen', zorder=101, s=.5)
    if obvious_anomaly_positions is not None:
        y = recon[obvious_anomaly_positions]
        ax.scatter(obvious_anomaly_positions, y, color='purple', label='obvious anomaly', zorder=102, s=.5)
    if predicted_anomaly_positions is not None and labels is not None and False:
        highest = np.max(np.concatenate((gt, recon)))
        lowest = np.min(np.concatenate((gt, recon)))
        hit_line_height = (highest - lowest) / 3. * 2. + lowest
        miss_line_height = (highest - lowest) / 3. + lowest
        hit_positions = np.intersect1d(np.where(labels == 1.), predicted_anomaly_positions)
        # print(hit_positions)
        miss_positions = np.setdiff1d(np.where(labels == 1.), predicted_anomaly_positions)
        hit_y = np.zeros(hit_positions.shape[0]) + hit_line_height
        miss_y = np.zeros(miss_positions.shape[0]) + miss_line_height
        ax.plot(hit_positions, hit_y, label='hitted', zorder=2, marker='.', linewidth=.5, markerfacecolor="None",
                markeredgewidth=.5,markersize=3)
        ax.plot(miss_positions, miss_y, label='missed', zorder=3, marker='.', linewidth=.5, markerfacecolor="None",
                markeredgewidth=.5,markersize=3)
    ax.legend()

    if save_file is None:
        fig.show()
    else:
        fig.savefig(save_file, dpi=600)
    plt.close(fig)


def cal_metrics(gt, predicted, total):
    gt_oz = np.zeros(total, dtype=float)
    gt_oz[gt] += 1.
    pred_oz = np.zeros(total, dtype=float)
    pred_oz[predicted] += 1.
    tp = np.where((pred_oz == 1) & (gt_oz == 1), 1., 0.).sum()
    fp = np.where((pred_oz == 1) & (gt_oz == 0), 1., 0.).sum()
    tn = np.where((pred_oz == 0) & (gt_oz == 0), 1., 0.).sum()
    fn = np.where((pred_oz == 0) & (gt_oz == 1), 1., 0.).sum()
    print(tp, fp, tn, fn)
    print(gt_oz.sum(), pred_oz.sum())
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    return recall, precision, f1

# python main.py -g 2,3,4,5 -GP -p 5 --figfile save/mse.png -e 5
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', "--gpu", action="store_true")
    parser.add_argument('-N', '--normalize_data', action='store_true')
    parser.add_argument('-P', '--parallel', action='store_true')
    parser.add_argument('-a', '--anomaly_ratio', default=0.05, type=float)
    parser.add_argument("-b", "--batch_size", default=1024, type=int)
    parser.add_argument("-e", "--epoch", default=100, type=int)
    parser.add_argument('-f', '--figfile', default=None)
    parser.add_argument("-g", "--gpu_device", default="0", type=str)
    parser.add_argument('-l', "--latent", default=5, type=int)
    parser.add_argument("-r", "--learning_rate", default=0.01, type=float)
    parser.add_argument('-p', '--process', default=None, type=int)
    parser.add_argument("-w", "--window_size", default=20, type=int)
    parser.add_argument('--which_set',default='1-1',type=str)
    parser.add_argument('--ICVAE',action='store_true')
    args = parser.parse_args()

    latent_size = args.latent
    anomaly_ratio = args.anomaly_ratio
    gpu = args.gpu
    learning_rate = args.learning_rate
    epoch = args.epoch
    batch_size = args.batch_size
    window_size = args.window_size
    gpu_device = args.gpu_device
    normalize = args.normalize_data
    process = args.process
    parallel = args.parallel
    figfile = args.figfile
    ICVAE=args.ICVAE

    which_set=args.which_set

    data_dir = '/remote-home/liuwenbo/pycproj/tsdata/data'
    dataset = 'smd'
    map_dir = '/remote-home/liuwenbo/pycproj/tsdata/maps/npmap'
    map = 'machine-'+which_set+'.npmap.pkl'
    train_set_file = os.path.join(data_dir, dataset, 'train/machine-'+which_set+'.pkl')
    test_set_file = os.path.join(data_dir, dataset, 'test/machine-'+which_set+'.pkl')
    label_file = os.path.join(data_dir, dataset, 'label/machine-'+which_set+'.pkl')
    map_file = os.path.join(map_dir, map)

    dataloader = DataLoader(train_set_file, test_set_file, label_file, normalize=normalize)
    dataloader.prepare_data(map_file, cnn_window_size=window_size, vae_window_size=window_size)
    # dataloader.draw_train_set()

    ivae = IVAE(dataloader, latent_size, gpu, learning_rate, gpu_device)
    icnn = ICNN(dataloader, window_size, gpu, learning_rate, gpu_device)
    icnn.train(epoch, batch_size, gpu)
    cnn_recon = icnn.infer(batch_size, gpu)
    if parallel:
        ivae.train_vaes_in_parallel(epoch, batch_size, gpu, proc=process)
        ivae_recon = None
    else:
        # ivae.train_cnns_in_serial(epoch,batch_size,gpu)
        ivae.train_vaes_in_serial(epoch, batch_size, gpu, figfile=figfile)
        ivae_recon = ivae.infer_in_serial(batch_size, gpu)

    cnn_ground_truth = dataloader.load_cnn_test_set_ground_truth()
    vae_ground_truth = dataloader.load_vae_test_set_ground_truth()
    labels = np.squeeze(dataloader.load_label_set())
    print(labels.shape)

    recon = np.concatenate((cnn_recon.transpose(), ivae_recon.transpose()), axis=0)
    ground_truth = np.concatenate((cnn_ground_truth.transpose(), vae_ground_truth.transpose()), axis=0)

    mse_list = np.mean((recon - ground_truth) ** 2, axis=0)
    obvious_abnormal_position = dataloader.load_obvious_anomaly_positions()
    obvious_abnormal_num = obvious_abnormal_position.shape
    anomaly_num = int(dataloader.load_test_set_size() * anomaly_ratio)
    suspicious_anomalies = np.setdiff1d(np.argsort(mse_list)[:anomaly_num], obvious_abnormal_position,
                                        assume_unique=True)
    # print(suspicious_anomalies)
    predicted_anomaly_positions = np.sort(
        np.concatenate((obvious_abnormal_position, suspicious_anomalies))[:anomaly_num])
    print('anomaly position dtype',predicted_anomaly_positions.dtype,suspicious_anomalies.dtype,obvious_abnormal_position.dtype)
    # print('pred an shape',predicted_anomaly_positions)

    # calculate scores
    recall, precision, f1 = cal_metrics(np.where(labels==1)[0], predicted_anomaly_positions.astype(int),
                                        dataloader.load_test_set_size())
    print('recall: %.3f, precision: %.3f, f1: %.3f' % (recall, precision, f1))

    # draw
    draw_gt_and_recon(ground_truth[0], recon[0], labels, predicted_anomaly_positions, obvious_abnormal_position,
                      'save/recon_gta.png')
    pool = mp.Pool()
    for i in range(ground_truth.shape[0]):
        pool.apply_async(draw_gt_and_recon,
                         args=(ground_truth[i], recon[i], labels, predicted_anomaly_positions,
                               obvious_abnormal_position, 'save/recon_gt' + str(i) + '.png'))
    pool.close()
    pool.join()
