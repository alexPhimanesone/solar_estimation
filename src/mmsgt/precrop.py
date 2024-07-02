import os
from os.path import join as opj
import sys
import numpy as np
from skimage import io
import pickle as pkl
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_processing')))
from crop_around_disk import crop_around_disk, get_disk_mask, get_disk_mask_list
from load import draw_id_pprad_list
from navig_dataset import get_id_pprad, get_id_endroit, get_id_mask

data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))
data_z_dir = "Z:/Work/Projects/solar_estimation/data/"
dataset_dir    = opj(data_dir      , "dataset/")
pics_dir       = opj(dataset_dir   , "pics")
dataset_z_dir  = opj(data_z_dir    , "dataset/")
pics_z_dir     = opj(dataset_z_dir , "pics_0606-1153/")
pprads_dir     = opj(dataset_dir   , "pprads/")
metadata_dir   = opj(dataset_dir   , "metadata/")
mmseg_orga_dir = opj(dataset_dir   , "mmseg_orga")
uncropped_dir  = opj(mmseg_orga_dir, "uncropped")
cropped_dir    = opj(mmseg_orga_dir, "cropped")


def precrop():
    
    '''
    # id_pprad_arr_fin

    # Load old id_pprad_arr
    id_pprad_arr = np.load(opj(dataset_dir, "mmseg_orga", "cropped", "id_pprad_arr.npy"))
    print(id_pprad_arr)

    # Build new one
    id_pprad_list_fin = draw_id_pprad_list()
    print(id_pprad_list_fin)
    id_pprad_arr_fin = np.zeros(len(id_pprad_list_fin))
    for i in range(len(id_pprad_list_fin)):
        if i < id_pprad_arr.shape[0]:
            id_pprad_arr_fin[i] = id_pprad_arr[i]
        else:
            id_pprad_arr_fin[i] = id_pprad_list_fin[i]
    print(id_pprad_arr_fin)
    print(id_pprad_arr_fin.shape)
    np.save(opj(dataset_dir, "mmseg_orga", "cropped", "id_pprad_arr_fin.npy"), id_pprad_arr_fin)
    '''

    #'''
    # Get id_pprad_list
    id_pprad_arr_path = opj(cropped_dir, "id_pprad_arr.npy")
    id_pprad_arr = np.load(id_pprad_arr_path)
    id_pprad_list = []
    for i in range(id_pprad_arr.shape[0]):
        id_pprad_list.append(int(id_pprad_arr[i]))
    print(id_pprad_list)
    #'''

    '''
    # img_dir
    subdir_list = ["train/", "test"]
    for subdir in subdir_list:
        dir = os.path.join(pics_dir, subdir)
        listdir = os.listdir(dir)
        for pic_name in listdir:
            #Get id_pic and pprad
            id_pic = pic_name[3:10]
            print(id_pic)
            id_pprad = id_pprad_list[int(get_id_endroit(id_pic=id_pic))]
            pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")
            # Crop
            pic = io.imread(os.path.join(dir, f"pic{id_pic}.jpg"))
            pic = crop_around_disk(pprad_path, pic)
            # Write pic
            io.imsave(opj(cropped_dir, "img_dir", subdir, f"{id_pic}.jpg"), pic)
    '''

    '''
    # ann_dir
    subdir_list = ["train/", "test"]
    for subdir in subdir_list:
        dir = os.path.join(uncropped_dir, "ann_dir", subdir)
        for mask_name in os.listdir(dir):
            #Get id_pic and pprad
            id_pic = mask_name[:7]
            print(id_pic)
            id_pprad = id_pprad_list[int(get_id_endroit(id_pic=id_pic))]
            pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")
            # Crop
            mask = io.imread(os.path.join(dir, f"{id_pic}.png"))
            mask = crop_around_disk(pprad_path, mask)
            # Write mask
            io.imsave(opj(cropped_dir, "ann_dir", subdir, f"{id_pic}.png"), mask)
    '''

print("START")
precrop()