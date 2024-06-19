import os
from os.path import join as opj
import sys
import numpy as np
from PIL import Image
sys.path.append("C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/")
sys.path.append("C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/data_processing/")
from crop_around_disk import crop_around_disk
from utils import read_raw_image
from navig_dataset import get_id_mask, get_id_endroit

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir    = opj(data_dir      , "dataset/")
pics_dir       = opj(dataset_dir   , "pics/")
masks_dir      = opj(dataset_dir   , "masks/")
pprads_dir     = opj(dataset_dir   , "pprads/")
metadata_dir   = opj(dataset_dir   , "metadata/")
mmseg_orga_dir = opj(dataset_dir   , "mmseg_orga")
uncropped_dir  = opj(mmseg_orga_dir, "uncropped")
cropped_dir    = opj(mmseg_orga_dir, "cropped")

palette = [[0, 0, 0], [255, 255, 255]]


'''
#PICS
for set_name in ['train', 'test']:
    set_name_set_dir = opj(mmseg_orga_dir, "img_dir", set_name)
    for pic_fn_old in os.listdir(set_name_set_dir):
        pic_fn_new = pic_fn_old[3:]
        os.rename(opj(set_name_set_dir, pic_fn_old,), opj(set_name_set_dir, pic_fn_new))
'''

# Get id_pprad_list
id_pprad_arr_path = opj(cropped_dir, "id_pprad_arr.npy")
id_pprad_arr = np.load(id_pprad_arr_path)
id_pprad_list = []
for i in range(id_pprad_arr.shape[0]):
    id_pprad_list.append(int(id_pprad_arr[i]))
print(id_pprad_list)

#'''
#MASKS
for set_name in ['train', 'test']:
    img_subdir = opj(cropped_dir, "img_dir", set_name)
    for pic_fn in os.listdir(img_subdir):
        
        # Get mask
        id_pic = pic_fn[:7]
        print(id_pic)
        id_mask = get_id_mask(id_pic=id_pic)
        mask = read_raw_image(opj(masks_dir, f"mask{id_mask}.raw"))
        
        # Crop
        id_pprad = id_pprad_list[int(get_id_endroit(id_pic=id_pic))]
        pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")
        mask = crop_around_disk(pprad_path, mask)

        # Format
        seg_img = Image.fromarray(np.squeeze(mask, axis=-1)).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))

        # Save
        seg_img.save(opj(cropped_dir, "ann_dir", set_name, f"{id_pic}.png"))
#'''