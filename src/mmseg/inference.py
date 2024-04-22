import os
import sys
import cv2
sys.path.append("C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/data_processing")
sys.path.append("C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/mmseg/configs")
from mmseg.apis import init_model, MMSegInferencer

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
checkpoints_dir = os.path.join(data_dir, "checkpoints/")
in_dir          = os.path.join(data_dir, "in/")
out_dir         = os.path.join(data_dir, "out/")
#configs_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/mmseg/configs/"


model = 'unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024'
'''
config_path = os.path.join(configs_dir, 'unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py')
checkpoint_path = os.path.join(checkpoints_dir,
                               "fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth")
'''


def inference(pics):
    #model = init_model(config_path, checkpoint_path, 'cpu')
    inferencer = MMSegInferencer(model=model)
    result = inferencer(pics)
    pred = result['predictions']
    return pred







'''
import numpy as np
import matplotlib.pyplot as plt
from crop_around_disk import crop_around_disk, get_disk_mask
from downsize_mask import min_pooling
from metrics import confusion_mat, acc, plot_cm

resolution = (512, 512)
id_pprad = 0 # le même pprad que celui sélectionné pour l'inférence
pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")

# load pred
pred = np.load(os.path.join(data_dir, "pred.npy"))
pred = np.where(pred == 10, 255, 0)

# load gt mask
height, width = get_height_width(id_pic)
mask = read_raw_image(os.path.join(masks_dir, f"mask{id_pic}.raw"), width, height)
mask = crop_around_disk(pprad_path, mask)
disk_mask = get_disk_mask(pprad_path)
mask, disk_mask_downsized = min_pooling(mask, resolution, disk_mask)

# Metrics
cm = confusion_mat(pred, mask, disk_mask_downsized)
accu = acc(pred, mask, disk_mask_downsized)

# Plots
nb_tot = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
for key in cm.keys():
    print(f'{key}: {cm[key] / nb_tot}')
print(f"acc: {accu}")
plot_cm(pred, mask, disk_mask_downsized)
'''
