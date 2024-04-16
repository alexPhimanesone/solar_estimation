import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/data_processing')))
from utils import read_raw_image, write_raw_image, path_raw_to_jpg, remove_first_zero
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
ee_endroit  = "09"
iii = "066"
id_pic = pp_phone + ee_endroit + iii


'''
#SAVE_CHECKS
from check import save_checks, pyplot
save_checks(id_pic)
#save_checks(id_pic, savenpz=True)
#pyplot()
'''

#'''
#CHECK_ENDROIT
from check import check_endroit
id_endroit = remove_first_zero(ee_endroit)
check_endroit(id_endroit)
#'''

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
endroit9
un peu de vent, extend là où ça a pas été extend
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
#PLOT_LINE
from edit_masks import plot_line
coo = 1430
axis = 'y'
plot_line(id_pic, coo, axis)
'''

'''
#PLOT_RECTANGLE
from edit_masks import plot_rectangle
y_start = 1085
y_end = 1110
x_start = 1258
x_end = 1271
plot_rectangle(id_pic, y_start, y_end, x_start, x_end, id_pic)
'''

'''
#PAINT_MASK
from edit_masks import paint_mask
y_start = 745
y_end = 770
x_start = 2915
x_end = 2939
paint_mask(id_pic, y_start, y_end, x_start, x_end, 'white', id_pic)
'''

'''
#AND_MASKS
from edit_masks import and_masks
mask1_path = os.path.join(masking_dir, "mask_th_blue_coth0008002.raw")
mask2_path = os.path.join(masking_dir, "mask_edge0008002.raw")
mask_and_path = os.path.join(masking_dir, "mask_edge_th0008002.raw")
and_masks(mask1_path, mask2_path, mask_and_path, id_pic, invert1=True, invert2=True, write_im=True)
'''

'''
#EXTEND_BLACK
from edit_masks import extend_black
id_mask = "0008002"
mask_path = os.path.join(masks_dir, f"mask{id_mask}.raw")
radius = 1
mask_extended_path = os.path.join(masking_dir, f"mask{radius}extended{id_mask}.raw")
extend_black(mask_path, mask_extended_path, id_pic, radius=radius, write_im=True)
'''
