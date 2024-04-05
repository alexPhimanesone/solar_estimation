import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml as yml
from crop_around_disk import get_disk, crop_around_disk
from utils import read_raw_image, read_csv

data_dir = "Z:/Work/Projects/solar_estimation/data/"
dataset_dir = os.path.join(data_dir, "dataset/")
pics_dir = os.path.join(dataset_dir, "pics/")
pprads_dir = os.path.join(dataset_dir, "pprads/")
masks_dir = os.path.join(dataset_dir, "masks/")
metadata_dir = os.path.join(dataset_dir, "metadata/")
checks_dir = os.path.join(data_dir, "checks/")


def save_checks(id_pic):

    NB_CLASSES = 3

    # Get paths
    pic_path = pics_dir + f'/pic{id_pic}.jpg'
    id_mask = eval(read_csv(os.path.join(metadata_dir, "pics_metadata.csv"), id_pic, "id_mask"))
    id_endroit = eval(read_csv(os.path.join(metadata_dir, "pics_metadata.csv"), id_pic, "id_endroit"))
    id_pprad = eval(read_csv(os.path.join(metadata_dir, "endroits_metadata.csv"), id_endroit, "id_pprad"))
    mask_path = masks_dir + f'/mask{id_mask}.raw'
    pprad_path = pprads_dir + f'/pprad{id_pprad}.yml'

    # Load images
    with open(pprad_path, 'r') as f:
        data = yml.load(f, Loader=yml.SafeLoader)
    principal_point = data['principal_point']
    radius = data['radius']
    pic_no_cropped = cv2.imread(pic_path)
    height_no_cropped, width_no_cropped, channel = pic_no_cropped.shape
    mask_no_cropped = read_raw_image(mask_path, width=width_no_cropped, height=height_no_cropped)
    pic = crop_around_disk(pprad_path, pic_no_cropped)
    mask = crop_around_disk(pprad_path, mask_no_cropped)

    # Compute checks
    white = np.array([255, 255, 255])
    black = np.array([0,   0,   0  ])
    red   = np.array([255, 0,   0  ])
    checks = {}
    checks['nonsky']    = np.where(np.all(mask == black, axis=-1, keepdims=True), pic, 0)
    checks['sky']       = np.where(np.all(mask == white, axis=-1, keepdims=True), pic, 0)
    checks['uncertain'] = np.where(np.all(mask == red  , axis=-1, keepdims=True), pic, 0)

    # Save checks
    cv2.imwrite(os.path.join(checks_dir, "img_original.png") , pic)
    cv2.imwrite(os.path.join(checks_dir, "nonsky0.png")      , checks['nonsky'])
    cv2.imwrite(os.path.join(checks_dir, "sky1.png")         , checks['sky'])
    cv2.imwrite(os.path.join(checks_dir, "uncertain2.png")   , checks['uncertain'])
    np.savez(os.path.join(checks_dir, "checks.npz"),
            img_origial=pic, nonsky0=checks['nonsky'], sky1=checks["sky"], uncertain2=checks['uncertain'])


def pyplot():
    """
    Plots the last npz computed.
    """
    checks = np.load(os.path.join(checks_dir, "checks.npz"))
    for key in checks.keys():
        plt.figure()
        plt.title(str(key))
        plt.imshow(np.clip(checks[key], 0, 255).astype(np.uint8), cmap='viridis')
    plt.show()
