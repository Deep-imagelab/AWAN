import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os


class HyperDatasetValid(udata.Dataset):
    def __init__(self, mode='valid'):
        if mode != 'valid':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Valid'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain1(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Train1'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain2(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Train2'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain3(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Train3'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain4(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Train4'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper
