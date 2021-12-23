import torch
import numpy as np
import hdf5storage
import time


def get_reconstruction_gpu(input, model):
    """As the limited GPU memory split the input."""
    model.eval()
    var_input = input.cuda()
    with torch.no_grad():
        start_time = time.time()
        var_output = model(var_input)
        end_time = time.time()

    return end_time-start_time, var_output.cpu()


def copy_patch1(x, y):
    x[:] = y[:]


def copy_patch2(stride, h, x, y):
    x[:, :, :, :-(h % stride)] = (y[:, :, :, :-(h % stride)] + x[:, :, :, :-(h % stride)]) / 2.0
    x[:, :, :, -(h % stride):] = y[:, :, :, -(h % stride):]


def copy_patch3(stride, w, x, y):
    x[:, :, :-(w % stride), :] = (y[:, :, :-(w % stride), :] + x[:, :, :-(w % stride), :]) / 2.0
    x[:, :, -(w % stride):, :] = y[:, :, -(w % stride):, :]


def copy_patch4(stride, w, h, x, y):
    x[:, :, :-(w % stride), :] = (y[:, :, :-(w % stride), :] + x[:, :, :-(w % stride), :]) / 2.0
    x[:, :, -(w % stride):, :-(h % stride)] = (y[:, :, -(w % stride):, :-(h % stride)] + x[:, :, -(w % stride):, :-(h % stride)]) /2.0
    x[:, :, -(w % stride):, -(h % stride):] = y[:, :, -(w % stride):, -(h % stride):]


def reconstruction_patch_image_gpu(rgb, model, patch, stride):
    all_time = 0
    _, _, w, h = rgb.shape
    rgb = torch.from_numpy(rgb).float()
    temp_hyper = torch.zeros(1, 31, w, h).float()
    # temp_rgb = torch.zeros(1, 3, w, h).float()
    for x in range(w//stride + 1):
        for y in range(h//stride + 1):
            if x < w // stride and y < h // stride:
                rgb_patch = rgb[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch]
                patch_time, hyper_patch = get_reconstruction_gpu(rgb_patch, model)
                # temp_hyper[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch] = hyper_patch
                copy_patch1(temp_hyper[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch], hyper_patch)
            elif x < w // stride and y == h // stride:
                rgb_patch = rgb[:, :, x * stride:x * stride + patch, -patch:]
                patch_time, hyper_patch = get_reconstruction_gpu(rgb_patch, model)
                # temp_hyper[:, :, x * stride:x * stride + patch, -patch:] = hyper_patch
                copy_patch2(stride, h, temp_hyper[:, :, x * stride:x * stride + patch, -patch:], hyper_patch)
            elif x == w // stride and y < h // stride:
                rgb_patch = rgb[:, :, -patch:, y * stride:y * stride + patch]
                patch_time, hyper_patch = get_reconstruction_gpu(rgb_patch, model)
                # temp_hyper[:, :, -patch:, y * stride:y * stride + patch] = hyper_patch
                copy_patch3(stride, w, temp_hyper[:, :, -patch:, y * stride:y * stride + patch], hyper_patch)
            else:
                rgb_patch = rgb[:, :, -patch:, -patch:]
                patch_time, hyper_patch = get_reconstruction_gpu(rgb_patch, model)
                # temp_hyper[:, :, -patch:, -patch:] = hyper_patch
                copy_patch4(stride, w, h, temp_hyper[:, :, -patch:, -patch:], hyper_patch)
            all_time += patch_time

    img_res = temp_hyper.numpy() * 1.0
    img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
    img_res_limits = np.minimum(img_res, 1.0)
    img_res_limits = np.maximum(img_res_limits, 0)
    return all_time, img_res_limits


def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)
