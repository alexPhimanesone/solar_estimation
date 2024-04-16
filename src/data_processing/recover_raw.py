import os
import numpy as np
import cv2
from utils import read_raw_image, get_height_width

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
masking_dir = os.path.join(data_dir   , "masking")
dataset_dir = os.path.join(data_dir   , "dataset/")
pics_dir    = os.path.join(dataset_dir, "pics/")

MAX_VALUE = 255

mask_path = os.path.join(masking_dir, "mask_th_blue0008002.raw")
pp_phone   = "00"
ee_endroit = "08"
iii_pic    = "002"


# Load
id_pic = pp_phone + ee_endroit + iii_pic
height, width = get_height_width(id_pic)
mask = read_raw_image(mask_path, width=width, height=height)

# Threshold
th = 179 #opti
_, mask_th = cv2.threshold(mask, th, MAX_VALUE, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(masking_dir, "mask_th_blue_coth0008002.jpg"), mask_th)


# Save
mask_th_c = np.ravel(mask_th, order='C')
with open(os.path.join(masking_dir, "mask_th_blue_coth00008002.raw"), 'wb') as f:
    f.write(mask_th_c)
