import os
from os.path import join as opj
import sys
import numpy as np
import torch
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/mmseg')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/ai')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/sky_detection')))
from losses import construct_loss
from utils import read_raw_image, write_raw_image, path_raw_to_jpg, get_model_path

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
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

#'''
#GET_SCORE
from inference import get_cm_arr
print("get_score starts running")
cm_arr = get_cm_arr()
np.save(opj(training_dir, "0610-1109", "cm_arr.npy"), cm_arr)
#'''


'''
#COMPUTE_METRICS
from inference import compute_metrics
cm_arr = np.load(opj(training_dir, "0610-1109", "cm_arr.npy"))
print(np.unique(cm_arr))
precision, recall, f1_score, global_accuracy = compute_metrics(cm_arr)
print(f'Total Precision: {precision:.4f}')
print(f'Total Recall: {recall:.4f}')
print(f'Total F1 Score: {f1_score:.4f}')
print(f'Total Global Accuracy: {global_accuracy:.4f}')
'''


'''
#SEE_PREDS
from inference import see_preds
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
pp_phone  = "00"
ee_endroit  = "01"
iii = "045"
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
from navig_dataset import get_id_pic_list
in_dir = os.path.join(data_dir, "in/")
files_names = os.listdir(in_dir)
id_pic_list = []
for file_name in files_names:
    id_pic_list.append(file_name[3:10])
save_checks_mult(id_pic_list, "in")
'''

'''
endroit0 ok
endroit1 ok
endroit2 ok
endroit3 ok
endroit4 ok
endroit5 ok
endroit6 ok
endroit7 ok
endroit8 ok
endroit9 ok
endroit10 2 37 ok
endroit10 39 113 ok
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
y_min =  485
x_min = 1000
y_start = 1153 + y_min
y_end   = 1181 + y_min
x_start =  343 + x_min
x_end   =  403 + x_min
plot_rectangle(id_pic, y_start, y_end, x_start, x_end)
'''

'''
#PAINT_MASK
from edit_masks import paint_mask
y_min =  485
x_min = 1000
y_starts = [1153 + y_min]
y_ends   = [1181 + y_min]
x_starts = [ 343 + x_min]
x_ends   = [ 403 + x_min]
mask_to_paint_path = os.path.join(masks_dir, "mask0010106.raw")
mask_painted_path  = os.path.join(masking_dir, "mask_painted0010106.raw")
paint_mask(mask_to_paint_path, mask_painted_path, id_pic, 'black',
           y_starts, y_ends, x_starts, x_ends, show_im=True)
'''

'''
#AND_MASKS
from edit_masks import and_masks
mask1_path = os.path.join(masking_dir, "mask_edge0010061.raw")
mask2_path = os.path.join(masks_dir, "mask0010106.raw")
mask_and_path = os.path.join(masking_dir, "mask_and0010106.raw")
and_masks(mask1_path, mask2_path, mask_and_path, id_pic, invert1=True, invert2=True, show_im=True)
'''

'''
#PLOT_LINE
from edit_masks import plot_line
coo = 1587
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
id_mask = "0010002"
radius = 4
y_start =  978
y_end   = 1100
x_start = 2144
x_end   = 2288
mask_path = os.path.join(masking_dir, f"mask4extended1extended1extended{id_mask}.raw")
mask_extended_path = os.path.join(masking_dir, f"mask{radius}extended4extended1extended1extended{id_mask}.raw")
extend_black(mask_path, mask_extended_path, id_pic, radius=radius,
             y_start=y_start, y_end=y_end, x_start=x_start, x_end=x_end, show_im=True)
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
