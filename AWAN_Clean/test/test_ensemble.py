import os
import numpy as np
import h5py
from utils.utils import save_matv73
import glob


mat_path0 = './test_results1/'
mat_path1 = './test_results2/'
mat_path2 = './test_results3/'
mat_path3 = './test_results4/'

save_path = './final_test_results/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
mat_path0_name = glob.glob(os.path.join(mat_path0, '*.mat'))
mat_path1_name = glob.glob(os.path.join(mat_path1, '*.mat'))
mat_path2_name = glob.glob(os.path.join(mat_path2, '*.mat'))
mat_path3_name = glob.glob(os.path.join(mat_path3, '*.mat'))
mat_path0_name.sort()
mat_path1_name.sort()
mat_path2_name.sort()
mat_path3_name.sort()

for i in range(len(mat_path1_name)):
    hf0 = h5py.File(mat_path0_name[i])
    data0 = hf0.get('cube')
    res0 = np.transpose(np.array(data0), [2, 1, 0])

    hf1 = h5py.File(mat_path1_name[i])
    data1 = hf1.get('cube')
    res1 = np.transpose(np.array(data1), [2, 1, 0])

    hf2 = h5py.File(mat_path2_name[i])
    data2 = hf2.get('cube')
    res2 = np.transpose(np.array(data2), [2, 1, 0])

    hf3 = h5py.File(mat_path3_name[i])
    data3 = hf3.get('cube')
    res3 = np.transpose(np.array(data3), [2, 1, 0])

    res = 0.25 * res0 + 0.25 * res1 + 0.25 * res2 + 0.25 * res3

    print(mat_path0_name[i].split('/')[-1], mat_path1_name[i].split('/')[-1], mat_path2_name[i].split('/')[-1], mat_path3_name[i].split('/')[-1])

    mat_dir = os.path.join(save_path, mat_path1_name[i].split('/')[-1])
    save_matv73(mat_dir, 'cube', res)


