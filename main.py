import argparse
import multiprocessing as mp
import os
import pickle
import platform

import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
from torch import optim

from DataLoader import DataLoader
from ILSTMCVAE import ILSTMCVAE
from VAE import VAE
from IVAE import IVAE
from ICNN import ICNN
from ICVAE import ICVAE

import numpy as np


def draw_gt_and_recon(gt, recon, labels=None, predicted_anomaly_positions=None, obvious_anomaly_positions=None,
                      save_file=None, dpi=300, title=None):
    # print(gt.shape,recon.shape)
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    fig.set_dpi(dpi)
    x = np.arange(gt.shape[0])
    ax.plot(x, gt, label='ground truth', zorder=0, linewidth=1)
    ax.plot(x, recon, alpha=0.7, label='reconstruction', zorder=1, linewidth=1)
    if labels is not None:
        x_labels = np.where(labels == 1)
        y_labels = gt[x_labels]
        ax.scatter(x_labels, y_labels, color='red', label='true anomaly', zorder=100, s=.5)
    if predicted_anomaly_positions is not None:
        y = recon[predicted_anomaly_positions]
        ax.scatter(predicted_anomaly_positions, y, label='pred anomaly', color='darkgreen', zorder=101, s=.5, alpha=0.7)
    if obvious_anomaly_positions is not None:
        y = recon[obvious_anomaly_positions]
        ax.scatter(obvious_anomaly_positions, y, color='purple', label='obvious anomaly', zorder=102, s=.5, alpha=0.7)
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
                markeredgewidth=.5, markersize=3)
        ax.plot(miss_positions, miss_y, label='missed', zorder=3, marker='.', linewidth=.5, markerfacecolor="None",
                markeredgewidth=.5, markersize=3)
    ax.legend(loc='lower right')
    if title is not None:
        ax.set_title(title)

    if save_file is None:
        fig.show()
    else:
        fig.savefig(save_file, dpi=dpi)
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
    parser.add_argument('-A', '--auto_anomaly_ratio', action='store_true')
    parser.add_argument('-B', '--batch_norm', action='store_true')
    parser.add_argument('-D', '--draw', action='store_true')
    parser.add_argument('-G', "--gpu", action="store_true")
    parser.add_argument('-N', '--normalize_data', action='store_true')
    parser.add_argument('-M', '--draw_mse', action='store_true')
    parser.add_argument('-P', '--parallel', action='store_true')
    parser.add_argument('-S', '--absolute_noise', action='store_true')
    parser.add_argument('-T', '--infer_train_set', action='store_true')
    parser.add_argument('-a', '--anomaly_ratio', default=0.05, type=float,
                        help='smd: 0.05, swat: 0.1214, wadi.old: 0.0384, wadi.new: 0.0577')
    parser.add_argument("-b", "--batch_size", default=1024, type=int)
    parser.add_argument("-c", "--cnn_window_size", default=20, type=int)
    parser.add_argument('-d', '--dpi', default=300, type=int)
    parser.add_argument("-e", "--epoch", default=100, type=int)
    parser.add_argument('-f', '--figfile', default=None)
    parser.add_argument("-g", "--gpu_device", default="0", type=str)
    parser.add_argument('-l', "--latent", default=5, type=int)
    parser.add_argument("-r", "--learning_rate", default=0.01, type=float)
    parser.add_argument('-p', '--process', default=None, type=int)
    parser.add_argument('-s', '--save_dir', default='save/tmp', type=str)
    parser.add_argument('-v', '--vae_window_size', default=1, type=int)
    parser.add_argument('--dataset', default='smd', type=str)
    parser.add_argument('--which_set', default='1-1', type=str)
    parser.add_argument('--which_model', default='vae', type=str)
    args = parser.parse_args()
    print(args)

    latent_size = args.latent
    draw = args.draw
    batch_norm = args.batch_norm
    gpu = args.gpu
    learning_rate = args.learning_rate
    epoch = args.epoch
    batch_size = args.batch_size
    cnn_window_size = args.cnn_window_size
    gpu_device = args.gpu_device
    normalize = args.normalize_data
    process = args.process
    parallel = args.parallel
    figfile = args.figfile
    which_model = args.which_model
    save_dir = args.save_dir
    which_set = args.which_set
    vae_window_size = args.vae_window_size
    dpi = args.dpi
    draw_mse = args.draw_mse and draw
    dataset = args.dataset
    auto_anomaly_ratio = args.auto_anomaly_ratio
    infer_train_set = args.infer_train_set
    absolute_noise = args.absolute_noise
    if auto_anomaly_ratio:
        if dataset == 'smd':
            anomaly_ratio = 0.05
        elif dataset == 'swat':
            anomaly_ratio = 0.1214
        elif dataset == 'wadi.old':
            anomaly_ratio = 0.0384
        elif dataset == 'wadi.new':
            anomaly_ratio = 0.0577
        else:
            anomaly_ratio = None
            print('invalid dataset')
            exit()
    else:
        anomaly_ratio = args.anomaly_ratio

    print('\033[0;34m program begin \033[0m')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if platform.system() == 'Windows':
        data_dir = 'E:\\Pycharm Projects\\causal.dataset\\data'
        map_dir = 'E:\\Pycharm Projects\\causal.dataset\\maps\\npmap'
    else:
        data_dir = '/remote-home/liuwenbo/pycproj/tsdata/data'
        map_dir = '/remote-home/liuwenbo/pycproj/tsdata/maps/npmap'

    if dataset == 'smd':
        map = 'machine-' + which_set + '.npmap.pkl'
        train_set_file = os.path.join(data_dir, dataset, 'train/machine-' + which_set + '.pkl')
        test_set_file = os.path.join(data_dir, dataset, 'test/machine-' + which_set + '.pkl')
        label_file = os.path.join(data_dir, dataset, 'label/machine-' + which_set + '.pkl')
        dataloader = DataLoader(train_set_file, test_set_file, label_file, normalize=normalize)
    elif dataset == 'wadi.old':
        pack_file = os.path.join(data_dir, dataset, 'raw.data.pkl')
        map = 'wadi.old.npmap.pkl'
        dataloader = DataLoader(normalize=normalize, pack_file=pack_file)
    elif dataset == 'wadi.new':
        pack_file = os.path.join(data_dir, dataset, 'raw.data.pkl')
        map = 'wadi.new.npmap.pkl'
        dataloader = DataLoader(normalize=normalize, pack_file=pack_file)
    elif dataset == 'swat':
        pack_file = os.path.join(data_dir, dataset, 'raw.data.pkl')
        map = 'swat.npmap.pkl'
        dataloader = DataLoader(normalize=normalize, pack_file=pack_file)
    else:
        dataloader = None
        map = None
        print('data unavaliable')
        exit()

    map_file = os.path.join(map_dir, map)
    dataloader.prepare_data(map_file, cnn_window_size=cnn_window_size, vae_window_size=vae_window_size)
    # dataloader.draw_train_set()

    ivae_train_recon = None
    if which_model == 'cvae':
        print('\033[0;35m using cvae \033[0m')
        icvae = ICVAE(dataloader, latent_size, gpu, learning_rate, gpu_device)
        icvae.train_vaes_in_serial(epoch, batch_size, gpu)
        ivae_recon = icvae.infer_in_serial(batch_size, gpu)
        if infer_train_set:
            ivae_train_recon = icvae.infer_in_serial_train_set(batch_size, gpu).transpose()
    elif which_model == 'lstmcvae':
        print('\033[0;35m using lstmcvae \033[0m')
        ilstmcvae = ILSTMCVAE(dataloader, latent_size, gpu, learning_rate, gpu_device)
        ilstmcvae.train_vaes_in_serial(epoch, batch_size, gpu)
        ivae_recon = ilstmcvae.infer_in_serial(batch_size, gpu)
        if infer_train_set:
            ivae_train_recon = ilstmcvae.infer_in_serial_train_set(batch_size, gpu).transpose()
    else:
        print('\033[0;35m using vae \033[0m')
        ivae = IVAE(dataloader, latent_size, gpu, learning_rate, gpu_device, batch_norm)
        if parallel:
            ivae.train_vaes_in_parallel(epoch, batch_size, gpu, proc=process)
        else:
            ivae.train_vaes_in_serial(epoch, batch_size, gpu, figfile=figfile)
        ivae_recon = ivae.infer_in_serial(batch_size, gpu)
        if infer_train_set:
            ivae_train_recon = ivae.infer_in_serial_train_set(batch_size, gpu).transpose()

    icnn = ICNN(dataloader, cnn_window_size, gpu, learning_rate, gpu_device)
    try:
        icnn.train(epoch, batch_size, gpu)
        cnn_recon = icnn.infer(batch_size, gpu)
        cnn_train_recon = icnn.infer_train_set(batch_size, gpu).transpose()
    except Exception as e:
        print('\033[0;35m', e, '\033[0m')
        cnn_recon = None
        cnn_train_recon = None

    cnn_ground_truth = dataloader.load_cnn_test_set_ground_truth()
    if which_model == 'vae':
        vae_ground_truth = dataloader.load_vae_test_set_ground_truth()
    else:
        vae_ground_truth = dataloader.load_cvae_test_ground_truth()
    labels = np.squeeze(dataloader.load_label_set())
    print(labels.shape)

    try:
        print('-------------------------')
        print('\033[0;35m', type(ivae_recon), type(cnn_recon), '\033[0m')
        recon = np.concatenate((cnn_recon.transpose(), ivae_recon.transpose()), axis=0)
        ground_truth = np.concatenate((cnn_ground_truth.transpose(), vae_ground_truth.transpose()), axis=0)
    except Exception as E:
        print('\033[0;36mexception occurred', E, '\033[0m')
        recon = ivae_recon.transpose()
        ground_truth = vae_ground_truth.transpose()

    mse_list = np.max((recon - ground_truth) ** 2, axis=0)
    mse_matrix_sorted = np.sort(((recon - ground_truth) ** 2), axis=1)
    mse_matrix = ((recon - ground_truth) ** 2)
    print('mse mat', mse_matrix_sorted.shape)
    if draw:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.imshow(mse_matrix_sorted)
        ax1.set_aspect(1000)
        ax2.imshow(mse_matrix)
        ax2.set_aspect(1000)
        fig.savefig(os.path.join(save_dir, 'mse.mat.sort.png'), dpi=dpi)
        plt.close(fig)

    if normalize:
        test_std, test_mean = dataloader.load_test_set_norm_params()
        print(recon.shape, test_std.shape, test_mean.shape)
        recon = recon * test_std + test_mean
        ground_truth = ground_truth * test_std + test_mean

    if absolute_noise:
        err_list = np.absolute(recon - ground_truth)
    else:
        err_list = (recon - ground_truth)
    print('absolute err: ')
    print(err_list.shape)
    print(np.max(err_list), np.min(err_list))
    mean_list = np.mean(err_list, axis=1)
    std_list = np.std(err_list, axis=1)
    prob_density_list = []
    for i in range(err_list.shape[0]):
        prob_density_list.append(norm(loc=mean_list[i], scale=std_list[i]).pdf(err_list[i]))
    prob_density_list = np.array(prob_density_list)
    print('prob density list shape', prob_density_list.shape)

    # obvious_abnormal_position = dataloader.load_obvious_abnormal_positions()
    # anomaly_score_list=mse_list
    anomaly_score_list = 1 - np.min(prob_density_list, axis=0)
    print('anomaly score list shape', anomaly_score_list.shape)
    obvious_abnormal_position = np.array([0, ])
    obvious_abnormal_num = obvious_abnormal_position.shape
    print('\033[0;33mobviously anomaly num\033[0m', obvious_abnormal_num)
    anomaly_num = int(dataloader.load_test_set_size() * anomaly_ratio)
    suspicious_anomalies = np.setdiff1d((np.argsort(anomaly_score_list)[::-1])[:anomaly_num], obvious_abnormal_position,
                                        assume_unique=True)
    print('\033[0;33msuspicous anomalies\033[0m', suspicious_anomalies.shape)
    print('\033[0;33mmse list sorted\033[0m', anomaly_score_list[suspicious_anomalies].shape)
    predicted_anomaly_positions = np.sort(
        np.concatenate((obvious_abnormal_position, suspicious_anomalies))[:anomaly_num])
    print('anomaly position dtype', predicted_anomaly_positions.dtype, suspicious_anomalies.dtype,
          obvious_abnormal_position.dtype)
    # print('pred and shape',predicted_anomaly_positions)

    # calculate scores
    recall, precision, f1 = cal_metrics(np.where(labels == 1)[0], predicted_anomaly_positions.astype(int),
                                        dataloader.load_test_set_size())
    mse = np.mean((recon - ground_truth) ** 2)
    # mse_each_dim = np.squeeze(np.mean((recon - ground_truth) ** 2, axis=1))
    # print('mse each dim',mse_each_dim)
    print('\033[0;31m', 'recall: %.3f, precision: %.3f, f1: %.3f, mse: %.5f' % (recall, precision, f1, mse),
          '\033[0m')

    print('recall: %.3f, precision: %.3f, f1: %.3f' % (recall, precision, f1), args,
          file=open(os.path.join(save_dir, 'summary.txt'), 'w'), sep='\n')

    # draw
    cnn_num = dataloader.non_constant_var_num - dataloader.load_vae_num()
    if draw:
        draw_gt_and_recon(ground_truth[0], recon[0], labels, predicted_anomaly_positions, obvious_abnormal_position,
                          os.path.join(save_dir, 'recon_gta.png'), dpi)
        pool = mp.Pool()
        for i in range(ground_truth.shape[0]):
            if i < cnn_num:
                title = 'cnn'
            else:
                title = 'lstm'
            pool.apply_async(draw_gt_and_recon, args=(ground_truth[i], recon[i], labels, predicted_anomaly_positions,
                                                      obvious_abnormal_position,
                                                      os.path.join(save_dir, 'recon_gt' + str(i) + '.png'), dpi, title))
        pool.close()
        pool.join()
        print('test recon done')

        fig, (ax1, ax2) = plt.subplots(2, 1)
        # abnormal_point = np.zeros(ground_truth.shape[1])
        # abnormal_point[np.where(labels == 1)] = 1.
        # abnormal_point = abnormal_point.reshape(1, -1).repeat(200, axis=0)
        ax1.imshow(ground_truth, label='train')
        ax2.imshow(recon, label='test')
        aspect_ratio = 200
        ax1.set_aspect(aspect_ratio)
        ax2.set_aspect(aspect_ratio)
        # ax3.imshow(abnormal_point)
        ax1.set_title('test ground truth')
        ax2.set_title('test recon')
        # ax3.set_title('abnormal points')
        ax1.set_yticks([])
        ax2.set_yticks([])
        # ax3.set_yticks([])
        fig.tight_layout()
        fig.set_dpi(dpi)
        fig.savefig(os.path.join(save_dir, 'test.and.recon.fig.png'), dpi=dpi)
        plt.close(fig)

        if infer_train_set:
            pool = mp.Pool()
            if cnn_train_recon is not None:
                train_set_recon = np.concatenate((cnn_train_recon, ivae_train_recon), axis=0)
            else:
                train_set_recon = ivae_train_recon
            train_set_ground_truth = dataloader.load_train_set_ground_truth()
            if normalize:
                train_std, train_mean = dataloader.load_train_set_norm_params()
                # train_std = train_std[dataloader.root_var.shape[0]:]
                # train_mean = train_mean[dataloader.root_var.shape[0]:]
                train_set_ground_truth = train_set_ground_truth * train_std + train_mean
                train_set_recon = train_set_recon * train_std + train_mean

            print('\033[0;33mtrain set ground truth shape \033[0m', train_set_ground_truth.shape[0])
            for i in range(train_set_ground_truth.shape[0]):
                if i < cnn_num:
                    title = 'cnn'
                else:
                    title = 'lstm'
                pool.apply_async(draw_gt_and_recon, args=(
                    train_set_ground_truth[i], train_set_recon[i], None, None, None,
                    os.path.join(save_dir, 'train_recon_gt' + str(i) + '.png'), dpi, title))
            pool.close()
            pool.join()
            print('train recon done')

            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(train_set_ground_truth, label='train')
            ax2.imshow(train_set_recon, label='test')
            ax1.set_aspect(aspect_ratio)
            ax2.set_aspect(aspect_ratio)
            ax1.set_title('train ground truth')
            ax2.set_title('train recon')
            ax1.set_yticks([])
            ax2.set_yticks([])
            fig.tight_layout()
            fig.set_dpi(dpi)
            fig.savefig(os.path.join(save_dir, 'train.and.recon.fig.png'), dpi=dpi)
            plt.close(fig)

            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow((train_set_ground_truth - train_set_recon) ** 2, label='train')
            ax2.imshow((ground_truth - recon) ** 2, label='test')
            ax1.set_aspect(aspect_ratio)
            ax2.set_aspect(aspect_ratio)
            ax1.set_title('train mse')
            ax2.set_title('test mse')
            ax1.set_yticks([])
            ax2.set_yticks([])
            fig.tight_layout()
            fig.set_dpi(dpi)
            fig.savefig(os.path.join(save_dir, 'mse.png'), dpi=dpi)
            plt.close(fig)
