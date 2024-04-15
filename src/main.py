import os
import sys
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/data_processing')))
from utils import read_raw_image, write_raw_image, path_raw_to_jpg
from navig_dataset import *

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
masking_dir  = os.path.join(data_dir   , "masking/")
dataset_dir  = os.path.join(data_dir   , "dataset/")
checks_dir   = os.path.join(data_dir   , "checks/")
pics_dir     = os.path.join(dataset_dir, "pics/")
pprads_dir   = os.path.join(dataset_dir, "pprads/")
masks_dir    = os.path.join(dataset_dir, "masks/")
metadata_dir = os.path.join(dataset_dir, "metadata/")
zoom_dir     = os.path.join(checks_dir , "zoom/")
channel_dir  = os.path.join(checks_dir , "channel/")

pp_phone  = "00"
ee_endroit  = "01"
iii = "018"
id_pic = pp_phone + ee_endroit + iii


#'''
from check import save_checks, pyplot
save_checks(id_pic)
#save_checks(id_pic, savenpz=True)
#pyplot()
#'''

'''
from check import check_endroit
id_endroit = ee_endroit.lstrip('0')
check_endroit(id_endroit)
'''

'''
from check import zoom
zoom(pp_phone, ee_endroit, x_start=8/32, x_end=10/32, y_start=14/32, y_end=15/32)
'''

'''
from check import check_channel
check_channel(id_pic)
'''

'''
from edit_masks import plot_line
coo = 1430
axis = 'y'
plot_line(id_pic, coo, axis)
'''

'''
from edit_masks import plot_rectangle
y_start = 1398
y_end = 1432
x_start = 1440
x_end = 1460
plot_rectangle(id_pic, y_start, y_end, x_start, x_end, id_pic)
'''

'''
from edit_masks import paint_mask
y_start = 1398
y_end = 1432
x_start = 1440
x_end = 1460
paint_mask(id_pic, y_start, y_end, x_start, x_end, 'white', id_pic)
'''

'''
pic_path = os.path.join(pics_dir, f"pic{id_pic}.jpg")
pic = cv2.imread(pic_path)
height, width = pic.shape[:2]
path = os.path.join(masking_dir, "mask_painted0009066.raw")
mask = read_raw_image(path, width=width, height=height)
cv2.imwrite(path_raw_to_jpg(path), mask)
'''

'''
from edit_masks import and_masks, join_masks
mask1_path = os.path.join(masking_dir, "th_blue_haut0009016.raw")
mask2_path = os.path.join(masking_dir, "th_blue_bas0009066.raw")
coo = 1587
axis = 'y'
join_masks(mask1_path, mask2_path, id_pic, coo, axis)
'''
