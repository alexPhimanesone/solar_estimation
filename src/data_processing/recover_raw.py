import os
import numpy as np
import cv2
from utils import read_raw_image

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
masking_dir = os.path.join(data_dir   , "masking")
dataset_dir = os.path.join(data_dir   , "dataset/")
pics_dir    = os.path.join(dataset_dir, "pics/")

MAX_VALUE = 255

mask1_path = os.path.join(masking_dir, "edges0004002.raw")
mask2_path = os.path.join(masking_dir, "thc1rr0004002.raw")
pp_phone   = "00" 
ee_endroit = "04"
iii_pic    = "002"


# Load
id_pic = pp_phone + ee_endroit + iii_pic
pic_path = os.path.join(pics_dir, f"pic{id_pic}.jpg")
pic = cv2.imread(pic_path)
height, width = pic.shape[:2]
mask1 = read_raw_image(mask1_path, width=width, height=height)
mask2 = read_raw_image(mask2_path, width=width, height=height)

# Threshold
th1 = 128
th2 = 200
_, mask1_th = cv2.threshold(mask1, th1, MAX_VALUE, cv2.THRESH_BINARY)
_, mask2_th = cv2.threshold(mask2, th2, MAX_VALUE, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(masking_dir, f"mask_edge{id_pic}.jpg"), mask1_th)
cv2.imwrite(os.path.join(masking_dir, f"mask_th{id_pic}.jpg")  , mask2_th)

# Save (ravel operations added a posteriori)
mask1_th_c = np.ravel(mask1_th, order='C')
mask2_th_c = np.ravel(mask2_th, order='C')
with open(os.path.join(masking_dir, f"mask_edge{id_pic}.raw"), 'wb') as f:
    f.write(mask1_th_c)
with open(os.path.join(masking_dir, f"mask_th{id_pic}.raw")  , 'wb') as f:
    f.write(mask2_th_c)
