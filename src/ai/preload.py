import os
import sys
import numpy as np
import torch
from skimage import io
import pickle as pkl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/data_processing')))
from crop_around_disk import crop_around_disk, get_disk_mask, get_disk_mask_list
from downsize import quantile_extraction
from load import draw_id_pprad_list
from navig_dataset import get_id_pprad, get_id_endroit, get_id_mask
from utils import read_raw_image, np_to_torch, write_raw_image, read_all_csv

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir   = os.path.join(data_dir     , "dataset/")
pics_dir      = os.path.join(dataset_dir  , "pics/")
masks_dir     = os.path.join(dataset_dir  , "masks/")
pprads_dir    = os.path.join(dataset_dir  , "pprads/")
metadata_dir  = os.path.join(dataset_dir  , "metadata/")
preloaded_dir = os.path.join(dataset_dir  , "preloaded/")
checks_dir    = os.path.join(data_dir     , "checks/")
inference_dir = os.path.join(data_dir     , "Inference/")
in_dir        = os.path.join(inference_dir, "in/")


def preload(resolution, qm):
    rqm_dir            = os.path.join(preloaded_dir, f"{resolution[0]}x{resolution[1]}_qm{qm}/")
    rqm_pics_dir       = os.path.join(rqm_dir      , "pics/")
    if not(os.path.exists(rqm_dir)):
        os.mkdir(rqm_dir)
    if not(os.path.exists(rqm_pics_dir)):
        os.mkdir(rqm_pics_dir)

    # id_pprad_arr
    id_pprad_list = draw_id_pprad_list()
    print(id_pprad_list)
    id_pprad_arr = np.zeros(len(id_pprad_list))
    for i in range(len(id_pprad_list)):
        id_pprad_arr[i] = id_pprad_list[i]
    print(id_pprad_arr.shape)
    np.save(os.path.join(rqm_dir, "id_pprad_arr.npy"), id_pprad_arr)

    # disk_mask_arr
    disk_mask_crop_list = get_disk_mask_list(id_pprad_list)
    disk_mask_crop_resize_arr = np.zeros((len(id_pprad_list), resolution[0], resolution[1]))
    for id_endroit in range(len(id_pprad_list)):
        print(id_endroit)
        # Load a pic
        data = read_all_csv(os.path.join(metadata_dir, "pics_metadata.csv"))
        for row in data:
            print(f"row['id_endroit']: {row['id_endroit']}")
            if row['id_endroit'] == str(id_endroit):
                id_pic = row['id_pic']
                break
        print(id_pic)
        for subdir in ["train/", "val/", "test/", "unmasked/"]:
            try:
                pic_path = os.path.join(pics_dir, subdir, f"pic{id_pic}.jpg")
                pic = io.imread(pic_path)
            except FileNotFoundError:
                continue
        # Crop and resize
        id_pprad = id_pprad_list[id_endroit]
        print(id_pprad)
        pic = crop_around_disk(os.path.join(pprads_dir, f"pprad{id_pprad}.yml"), pic)
        _, disk_mask_crop_resize = quantile_extraction(pic, resolution, disk_mask_crop_list[id_endroit], 1, 'higher')
        # Add to array
        disk_mask_crop_resize_arr[id_endroit] = disk_mask_crop_resize
    print(disk_mask_crop_resize_arr.shape)
    np.save(os.path.join(rqm_dir, "disk_mask_arr.npy"), disk_mask_crop_resize_arr)

    # mask_dict
    mask_dict = {}
    import cv2
    for mask_name in os.listdir(masks_dir):
        id_mask = mask_name[4:11]
        print(id_mask)
        mask = read_raw_image(os.path.join(masks_dir, mask_name))
        id_pprad = id_pprad_list[int(get_id_endroit(id_mask=id_mask))]
        pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")
        disk_mask = get_disk_mask(pprad_path)
        mask = crop_around_disk(pprad_path, mask)
        mask, _ = quantile_extraction(mask, resolution, disk_mask, qm/1000, 'lower')
        mask[mask == 255] = 1
        mask_dict[id_mask] = mask
    with open(os.path.join(rqm_dir, "mask_dict.pkl"), 'wb') as f:
        pkl.dump(mask_dict, f)

    # pics dir
    subdir_list = ["train/", "val/", "test"]
    for subdir in subdir_list:
        dir = os.path.join(pics_dir, subdir)
        for pic_name in os.listdir(dir):
            #Get id_pic and pprad
            id_pic = pic_name[3:10]
            print(id_pic)
            id_pprad = id_pprad_list[int(get_id_endroit(id_pic=id_pic))]
            pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")
            disk_mask = get_disk_mask(pprad_path)
            # Crop and downsize
            pic = io.imread(os.path.join(dir, f"pic{id_pic}.jpg"))
            pic = crop_around_disk(pprad_path, pic)
            pic, _= quantile_extraction(pic, resolution, disk_mask, 1, 'higher')
            # Write pic
            io.imsave(os.path.join(rqm_pics_dir, subdir, f"pic{id_pic}.png"), pic)
    