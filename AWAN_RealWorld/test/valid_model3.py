import torch
import os
import numpy as np
import cv2
from utils.AWAN import AWAN
import glob
from utils.utils import reconstruction_patch_image_gpu, save_matv73

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_path = './models/DRAB8_200_v3.pth'
result_path = './valid_results3/'
img_path = './NTIRE2020_Validation_RealWorld/'
var_name = 'cube'

# save results
if not os.path.exists(result_path):
    os.makedirs(result_path)
model = AWAN(3, 31, 200, 8)
save_point = torch.load(model_path)
model_param = save_point['state_dict']
model_dict = {}
for k1, k2 in zip(model.state_dict(), model_param):
    model_dict[k1] = model_param[k2]
model.load_state_dict(model_dict)
model = model.cuda()

img_path_name = glob.glob(os.path.join(img_path, '*.jpg'))
img_path_name.sort()

for i in range(len(img_path_name)):
    # load rgb images
    rgb = cv2.imread(img_path_name[i])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb) / 255.0
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    print(img_path_name[i].split('/')[-1])

    _, img_res = reconstruction_patch_image_gpu(rgb, model, 128, 128)
    _, img_res_overlap = reconstruction_patch_image_gpu(rgb[:, :, 128//2:, 128//2:], model, 128, 128)
    img_res[128//2:, 128//2:, :] = (img_res[128//2:, 128//2:, :] + img_res_overlap) / 2.0

    rgbFlip = np.flip(rgb, 2).copy()
    _, img_resFlip = reconstruction_patch_image_gpu(rgbFlip, model, 128, 128)
    _, img_res_overlapFlip = reconstruction_patch_image_gpu(rgbFlip[:, :, 128 // 2:, 128 // 2:], model, 128, 128)
    img_resFlip[128 // 2:, 128 // 2:, :] = (img_resFlip[128 // 2:, 128 // 2:, :] + img_res_overlapFlip) / 2.0
    img_resFlip = np.flip(img_resFlip, 0)
    img_res = (img_res + img_resFlip) / 2

    mat_name = img_path_name[i].split('/')[-1][:-14] + '.mat'
    mat_dir = os.path.join(result_path, mat_name)

    save_matv73(mat_dir, var_name, img_res)







