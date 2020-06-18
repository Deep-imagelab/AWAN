import torch
import numpy as np
import hdf5storage
import time


def get_reconstruction_cpu(input, model):
    model.eval()
    var_input = input
    with torch.no_grad():
        start_time = time.time()
        var_output = model(var_input)
        end_time = time.time()

    return end_time-start_time, var_output


def reconstruction_whole_image_cpu(rgb, model):
    all_time, img_res = get_reconstruction_cpu(torch.from_numpy(rgb).float(), model)
    img_res = img_res.cpu().numpy() * 1.0
    img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
    img_res_limits = np.minimum(img_res, 1.0)
    img_res_limits = np.maximum(img_res_limits, 0)
    return all_time, img_res_limits


def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)
