from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import hdf5storage


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
            'epoch': epoch,
            'iter': iteration,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
    
    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))


def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)


def record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()    
    loss_csv.close


class Loss_train(nn.Module):
    def __init__(self):
        super(Loss_train, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / label
        # error = torch.abs(outputs - label)
        rrmse = torch.mean(error.view(-1))
        return rrmse


class Loss_valid(nn.Module):
    def __init__(self):
        super(Loss_valid, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / label
        # error = torch.abs(outputs - label)
        rrmse = torch.mean(error.view(-1))
        return rrmse


class LossTrainCSS(nn.Module):
    def __init__(self):
        super(LossTrainCSS, self).__init__()
        self.model_hs2rgb = nn.Conv2d(31, 3, 1, bias=False)
        filtersPath = './cie_1964_w_gain.npz'
        cie_matrix = np.load(filtersPath)['filters']
        cie_matrix = torch.from_numpy(np.transpose(cie_matrix, [1, 0])).unsqueeze(-1).unsqueeze(-1).float()
        self.model_hs2rgb.weight.data = cie_matrix

    def forward(self, outputs, label, rgb_label):
        rrmse = self.mrae_loss(outputs, label)
        # hs2rgb
        with torch.no_grad():
            rgb_tensor = self.model_hs2rgb(outputs)
            rgb_tensor = rgb_tensor / 255
            rgb_tensor = torch.clamp(rgb_tensor, 0, 1) * 255
            # rgb_tensor = torch.tensor(rgb_tensor, dtype=torch.uint8)
            rgb_tensor = torch.tensor(rgb_tensor).byte().float()
            rgb_tensor = rgb_tensor / 255
        rrmse_rgb = self.rgb_mrae_loss(rgb_tensor, rgb_label)
        return rrmse, rrmse_rgb

    def mrae_loss(self, outputs, label):
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.view(-1))
        return mrae

    def rgb_mrae_loss(self, outputs, label):
        error = torch.abs(outputs - label)
        mrae = torch.mean(error.view(-1))
        return mrae

