import os
from os.path import join as opj
import sys
import inspect
import numpy as np
import torch
import json
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_processing')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mmsgt')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai')))
from utils import read_raw_image, write_raw_image, path_raw_to_jpg, get_model_path

data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
masking_dir   = os.path.join(data_dir     , "masking/")
dataset_dir   = os.path.join(data_dir     , "dataset/")
checks_dir    = os.path.join(data_dir     , "checks/")
pics_dir      = os.path.join(dataset_dir  , "pics/")
pprads_dir    = os.path.join(dataset_dir  , "pprads/")
masks_dir     = os.path.join(dataset_dir  , "masks/")
metadata_dir  = os.path.join(dataset_dir  , "metadata/")
preloaded_dir = os.path.join(dataset_dir  , "preloaded/")
zoom_dir      = os.path.join(checks_dir   , "zoom/")
channel_dir   = os.path.join(checks_dir   , "channel/")
training_dir  = os.path.join(data_dir     , "Training/")
inference_dir = os.path.join(data_dir     , "Inference/")
in_dir        = os.path.join(inference_dir, "in/")


#=================================================================
#                       SKY_DETECTION
#=================================================================

'''
#WRITE_PERF
from utils import get_device
from perf import write_perf

str_date_time = "0427-1704"
epoch = 92
subdir = 'train'

write_perf(get_device(), str_date_time, epoch, subdir)
'''

'''
#TRAIN
from train import train

# HYPERPARAM
hp = {}
hp['resolution']    = (512, 512)
hp['qm']            = 250
hp['batch_sizes']   = {'train': 16, 'val': 16, 'test': 16}
hp['alpha_leaky']   = 0.1
hp['loss']          = 'BCEWithLogits_disk'
hp['epochs']        = 200
hp['min_delta']     = 0
hp['initial_lr']    = 1e-2
hp['lr_patience']   = 5
hp['lr_factor']     = 0.8
hp['lr_min']        = 5e-6
hp['stop_patience'] = 70

train(get_device(), hp)
'''

'''
#PRELOAD
from preload import preload
resolution = (512, 512)
qm = 250
preload(resolution, qm)
'''

'''
#INSPECT_MASK_DICT
import pickle
import matplotlib.pyplot as plt
resolution = (512, 512)
qm = 250
rqm_dir = os.path.join(preloaded_dir, f"{resolution[0]}x{resolution[1]}_qm{qm}/")
with open(os.path.join(rqm_dir, "mask_dict.pkl"), 'rb') as f:
    mask_dict = pickle.load(f)
for key in mask_dict.keys():
    print(str(key))
    print(mask_dict[key].shape)
    print(np.unique(mask_dict[key]))
    plt.figure()
    plt.imshow(mask_dict[key])
    plt.show()
'''


#=================================================================
#                       MMSEGMENTATION
#=================================================================

'''
#VIS_TRAIN and VIS_VAL
from vis import vis_train, vis_val
timestamp = "0705-1214"
vis_train(timestamp)
vis_val(timestamp)
'''

# 0703-0155: iter = 900, best mAcc, 6550 2nd best, 6550=MAGIC
# 0704-1937: iter = 800, best mAcc,  960 2nd best
# 0705-1214: iter = 440, best mAcc, 1080 best aAcc

'''
#GET_SCORE
from inference import get_cm_arr, compute_metrics
print("get_score starts running")
cm_arr_path = opj(training_dir, "0703-0155", "cm_arr6550.npy")
cm_arr = get_cm_arr()
np.save(cm_arr_path, cm_arr)
#cm_arr = np.load(cm_arr_path)
print(cm_arr)
metrics = compute_metrics(cm_arr)
for key in metrics.keys():
    print(f"{key}: {metrics[key]}")
'''

'''
#SEE_PREDS
from mmsgt.inference import see_preds
print("see_preds starts running")
preds = see_preds()
'''

'''
#INSPECT_DIMS
from inference import inspect_dims
inspect_dims()
'''

'''
# INFERENCE
import matplotlib.pyplot as plt
import cv2
from crop_around_disk import crop_around_disk, get_disk_mask
from downsize import quantile_pooling
from metrics import confusion_mat_rates, acc, plot_cm
from inference import inference

inference_dir = os.path.join(data_dir, "inference/")
in_dir        = os.path.join(inference_dir, "in/")
out_dir       = os.path.join(inference_dir, "out/")

resolution = (128, 128)


for file_name in os.listdir(in_dir):

    # Load pic and mask
    pic_path = os.path.join(in_dir, file_name)
    id_pic = file_name[3:10]
    pic = cv2.imread(pic_path)
    id_mask = get_id_mask(id_pic=id_pic)
    mask_path = os.path.join(masks_dir, f"mask{id_mask}.raw")
    height, width = get_height_width(id_pic)
    mask = read_raw_image(mask_path, width=width, height=height)

    # Crop pic and mask
    id_pprad = get_id_pprad(id_pic=id_pic)
    pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")
    pic = crop_around_disk(pprad_path, pic)
    mask = crop_around_disk(pprad_path, mask)
    disk_mask = get_disk_mask(pprad_path)

    # Downsize pic and mask
    pic, disk_mask = quantile_pooling(pic , resolution, disk_mask, q=1  )
    mask, _        = quantile_pooling(mask, resolution, disk_mask, q=1/4)

    # Inference
    pred = inference(pic)
    pred = np.where(pred == 10, 255, 0)

    # Compute metrics
    metrics = {}
    metrics['cm'] = confusion_mat_rates(pred, mask, disk_mask)
    metrics['acc'] = acc(pred, mask, disk_mask)    

    # Save metrics
    metrics_json = json.dumps(metrics)
    metrics_path = os.path.join(out_dir, f"metrics{id_pic}.txt")
    with open(metrics_path, 'w') as f:
        f.write(metrics_json)
    fig_path = os.path.join(out_dir, f"fig{id_pic}.png")
    plot_cm(pred, mask, disk_mask, fig_path)
'''



#=================================================================
#                       MASK ANNOTATION
#=================================================================

'''
pp_phone  = "01"
ee_endroit  = "06"
iii = "014"
id_pic = pp_phone + ee_endroit + iii
'''

'''
#SAVE_CHECKS
from check import save_checks, pyplot
save_checks(id_pic)
#save_checks(id_pic, savenpz=True)
#pyplot()
'''

'''
#SAVE_CHECKS_MULT
from check import save_checks_mult
id_pic_list = []
for file_name in os.listdir(opj(dataset_dir, "pics")):
    id_pic = file_name[3:10]
    ppee = id_pic[:4]
    if ppee == "0106":
        id_pic_list.append(id_pic)
save_checks_mult(id_pic_list)
'''

'''
0000 ok
0001 ok
0002 ok
0003 ok
0004 ok
0005 ok
0006 ok
0007 ok
0008 ok
0009 ok
0010 ok
0010 ok
0011 ok
0012 ok
0100 ok
0101 ok
0102 ok
0103 ok
0104 ok
0105 ok
0106 ok
'''

'''
#ZOOM
from check import zoom
zoom(pp_phone, ee_endroit, x_start=8/32, x_end=10/32, y_start=14/32, y_end=15/32)
'''

'''
#CHECK_CHANNEL
from check import check_channel
check_channel(id_pic)
'''

'''
#PLOT_RECTANGLE
from edit_masks import plot_rectangle
y_min =  605
x_min = 1109
y_start = 1866 #+ y_min
y_end   = 1911 #+ y_min
x_start = 2775 #+ x_min
x_end   = 2792 #+ x_min
#plot_rectangle(id_pic, y_start, y_end, x_start, x_end)

import cv2
import matplotlib.pyplot as plt

# Load pic
mask_path = os.path.join(masks_dir, "mask0105028.raw")
mask = read_raw_image(mask_path)
mask = np.reshape(np.repeat(mask, 3, axis=-1), (3024, 4032, 3))
print(mask.shape)

# Create mask_rectangle
thickness = 1
pic_rectangle = cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 128, thickness)

# Show mask_rectangle
plt.figure()
plt.imshow(pic_rectangle)
plt.show()
'''

'''
#PAINT_MASK
from edit_masks import paint_mask
y_min =  605
x_min = 1109
y_starts = [1866]
y_ends   = [1911]
x_starts = [2775]
x_ends   = [2792]
mask_to_paint_path = os.path.join(masks_dir, "mask0105028.raw")
mask_painted_path  = os.path.join(masking_dir, "mask0105028paint.raw")
paint_mask(mask_to_paint_path, mask_painted_path, id_pic, 'black',
           y_starts, y_ends, x_starts, x_ends, show_im=True)
'''

'''
#AND_MASKS
from edit_masks import and_masks
id_pic1 = "0106014"
id_pic2 = "0106016"
mask1_path = os.path.join(masking_dir, f"mask{id_pic1}th.raw")
mask2_path = os.path.join(masking_dir, f"mask{id_pic2}edge.raw")
mask_and_path = os.path.join(masking_dir, f"mask{id_pic}and.raw")
and_masks(mask1_path, mask2_path, mask_and_path, id_pic, invert1=True, invert2=True, show_im=True)
'''

'''
#PLOT_LINE
from edit_masks import plot_line
coo = 1430
axis = 'y'
plot_line(id_pic, coo, axis)
'''

'''
#JOIN_MASKS
from edit_masks import join_masks
mask1_path = os.path.join(masking_dir, "mask_edge_th_haut0009016.raw")
mask2_path = os.path.join(masking_dir, "mask_edge_th_bas0009066.raw")
mask_and_path = os.path.join(masking_dir, "mask_edge_th0009066.raw")
coo = 1587
axis = 'y'
join_masks(mask1_path, mask2_path, mask_and_path, id_pic, coo, axis, write_im=True)
'''

'''
#EXTEND_BLACK
from edit_masks import extend_black
id_mask = id_pic
radius = 2
y_start =  1430
y_end   = 1100
x_start = 2144
x_end   = 2288
mask_path = os.path.join(masking_dir, f"mask{id_mask}and.raw")
mask_extended_path = os.path.join(masking_dir, f"mask{id_mask}andext{radius}.raw")
extend_black(mask_path, mask_extended_path, id_pic, radius=radius,
             show_im=True)
'''

'''
#PATCH_MASK
from edit_masks import patch_mask
y_start =    0
y_end   = 1430
x_start = 1504
x_end   = 2808
mask_to_patch_path = os.path.join(masks_dir  , "mask0009066.raw")
mask_patch_path    = os.path.join(masking_dir, "mask_edge_th_bas0009066.raw")
mask_patched_path  = os.path.join(masking_dir, "mask_patched0009066.raw")
patch_mask(mask_to_patch_path, mask_patch_path, mask_patched_path, id_pic,
           y_start, y_end, x_start, x_end, show_im=True)
'''



#=================================================================
#                       CALIBRATION
#=================================================================

#calib_set_path = opj(data_dir, "calibration", "calibs", "calib13")

'''
#CALIBRATE
from crop_around_disk import calibrate
calibrate(calib_set_path)
'''

'''
#ESTIMATE_RADIUS
from crop_around_disk import estimate_radius
estimate_radius(calib_set_path)
'''

'''
#CHECK_CROP
from crop_around_disk import check_crop
id_endroit = "18"
check_crop(id_endroit)
'''