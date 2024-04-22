import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml as yml
from crop_around_disk import crop_around_disk
from get_uncertain_mask import get_uncertain_mask
from utils import read_raw_image, read_csv
from navig_dataset import get_id_mask, get_id_endroit, get_id_pprad, get_id_pic_list

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

NB_CLASSES = 3
MAX_VAL = 255
white = np.array([255, 255, 255])
black = np.array([0,   0,   0  ])


def save_checks(id_pic, savenpz=False):

    # Get ids
    id_mask = get_id_mask(id_pic=id_pic)
    id_endroit = get_id_endroit(id_pic=id_pic)
    id_pprad = get_id_pprad(id_endroit=id_endroit)

    # Load pic and mask
    pic_path = pics_dir + f'/pic{id_pic}.jpg'
    pic_no_cropped = cv2.imread(pic_path)
    height_no_cropped, width_no_cropped, channel = pic_no_cropped.shape
    if id_mask != "-1":
        mask_path = masks_dir + f'/mask{id_mask}.raw'
        mask_no_cropped = read_raw_image(mask_path, width=width_no_cropped, height=height_no_cropped)
    else:
        print("This pic doesn't have a mask.")
        mask_no_cropped = np.zeros((height_no_cropped, width_no_cropped, 3))
    
    # Crop pic and mask
    pprad_path = pprads_dir + f'/pprad{id_pprad}.yml'
    pic = crop_around_disk(pprad_path, pic_no_cropped)
    mask = crop_around_disk(pprad_path, mask_no_cropped)

    # Get uncertain_mask
    th = 255
    print(f"temp th: {th}")
    uncertain_mask = get_uncertain_mask(pic, th)
    uncertain_mask = np.expand_dims(uncertain_mask, axis=-1)

    # Compute checks
    checks = {}
    checks['nonsky']    = np.where(np.logical_and(np.all(mask           == black, axis=-1, keepdims=True),
                                                  np.all(uncertain_mask == 0    , axis=-1, keepdims=True)),
                                   pic, 0)
    checks['sky']       = np.where(np.logical_and(np.all(mask           == white, axis=-1, keepdims=True),
                                                  np.all(uncertain_mask == 0    , axis=-1, keepdims=True)),
                                   pic, 0)
    checks['uncertain'] = np.where(               np.all(uncertain_mask == 255  , axis=-1, keepdims=True),
                                   0, pic)

    # Delete previous checks
    for file in os.listdir(checks_dir):
        if (file.startswith("0img") or file.startswith("0non") or
            file.startswith("1img") or file.startswith("1sky") or
            file.startswith("2img") or file.startswith("2unc")):
            os.remove(os.path.join(checks_dir, file))
    
    # Save checks
    cv2.imwrite(os.path.join(checks_dir, "0img_original"     + id_pic + ".png"), pic)
    cv2.imwrite(os.path.join(checks_dir, "0nonsky"       + id_pic + ".png"), checks['nonsky'])
    cv2.imwrite(os.path.join(checks_dir, "1img_original" + id_pic + ".png"), pic)
    cv2.imwrite(os.path.join(checks_dir, "1sky"          + id_pic + ".png"), checks['sky'])
    cv2.imwrite(os.path.join(checks_dir, "2img_original" + id_pic + ".png"), pic)
    cv2.imwrite(os.path.join(checks_dir, "2uncertain"    + id_pic + ".png"), checks['uncertain'])
    npz_path = os.path.join(checks_dir, "checks.npz")
    if os.path.exists(npz_path):
        os.remove(npz_path)
    if savenpz == True:
        np.savez(npz_path,
                 img_origial=pic, nonsky0=checks['nonsky'], sky1=checks["sky"], uncertain2=checks['uncertain'])


def save_checks_mult(id_pic_list, checks_mult_dir_name=None):
    """
    Save in a directory checks of multiple pics.
    Args:
    List of id_pic, corresponding to a mask or to an endroit.
    """

    # Get dims
    pic0_path = pics_dir + f'/pic{id_pic_list[0]}.jpg'
    pic0_no_cropped = cv2.imread(pic0_path)
    height_no_cropped, width_no_cropped = pic0_no_cropped.shape[:2]

    # Get pprad_path
    id_pprad = get_id_pprad(id_pic=id_pic_list[0])
    if id_pprad == "-1":
            print("This endroit doesn't have a pprad. Assigning a random pprad with matching phone/lens.")
            id_pprad = get_random_matching_pprad(id_pic=id_pic_list[0])
    pprad_path = pprads_dir + f'/pprad{id_pprad}.yml'

    if checks_mult_dir_name is None:
        checks_mult_dir_name = "checks_mult/"
    checks_mult_dir = os.path.join(checks_dir, checks_mult_dir_name)
    os.mkdir(checks_mult_dir)
    for id_pic in id_pic_list:
    
        # Load pic and mask
        pic_path = pics_dir + f'/pic{id_pic}.jpg'
        pic_no_cropped = cv2.imread(pic_path)
        id_mask = get_id_mask(id_pic=id_pic)
        if id_mask != "-1":
            mask_path = masks_dir + f'/mask{id_mask}.raw'
            mask_no_cropped = read_raw_image(mask_path, width=width_no_cropped, height=height_no_cropped)
        else:
            continue
        
        # Crop pic and mask
        pic  = crop_around_disk(pprad_path, pic_no_cropped)
        mask = crop_around_disk(pprad_path, mask_no_cropped)

        # Get uncertain_mask
        th = 255
        print(f"temp th: {th}")
        uncertain_mask = get_uncertain_mask(pic, th)
        uncertain_mask = np.expand_dims(uncertain_mask, axis=-1)

        # Compute checks
        checks = {}
        checks['nonsky']    = np.where(np.logical_and(np.all(mask           == black, axis=-1, keepdims=True),
                                                      np.all(uncertain_mask == 0    , axis=-1, keepdims=True)),
                                    pic, 0)
        checks['sky']       = np.where(np.logical_and(np.all(mask           == white, axis=-1, keepdims=True),
                                                      np.all(uncertain_mask == 0    , axis=-1, keepdims=True)),
                                    pic, 0)
        checks['uncertain'] = np.where(               np.all(uncertain_mask == 255  , axis=-1, keepdims=True),
                                    0, pic)
        
        # Save checks in checks_mult_dir
        cv2.imwrite(os.path.join(checks_mult_dir, id_pic + "0nonsky"    + ".png"), checks['nonsky'])
        cv2.imwrite(os.path.join(checks_mult_dir, id_pic + "1sky"       + ".png"), checks['sky'])
        cv2.imwrite(os.path.join(checks_mult_dir, id_pic + "2uncertain" + ".png"), checks['uncertain'])
        print(f"Pic {id_pic} saved at {checks_mult_dir}")


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


def zoom(pp_phone: str,ee_endroit: str, x_start=0.0, x_end=1.0, y_start=0.0, y_end=1.0):

    # Create pic_path_list
    pic_path_list = []
    file_list = os.listdir(pics_dir)
    for file_name in file_list:
        if file_name.startswith("pic" + pp_phone + ee_endroit):
            pic_path_list.append(file_name)

    # Create ranges
    pic = cv2.imread(os.path.join(pics_dir, pic_path_list[0]))
    W_origin = pic.shape[1]
    H_origin = pic.shape[0]
    y_start = int(y_start * H_origin)
    y_end   = int(y_end   * H_origin)
    x_start = int(x_start * W_origin)
    x_end   = int(x_end   * W_origin)

    # Delete all previous zoom pics 
    file_list = os.listdir(zoom_dir)
    for file_name in file_list:
        if file_name.startswith("zoom"):
            file_path = os.path.join(zoom_dir, file_name)
            os.remove(file_path)

    # Zoom and write
    for i in range(len(pic_path_list)):
        pic = cv2.imread(os.path.join(pics_dir, pic_path_list[i]))
        zoom = pic[y_start:y_end, x_start:x_end]
        zoom_path = f"zoom{pic_path_list[i][3:10]}.jpg"
        cv2.imwrite(os.path.join(zoom_dir, zoom_path), zoom)


def check_channel(id_pic):

    # Compute unicolor images
    pic_path = os.path.join(pics_dir, f"pic{id_pic}.jpg")
    pic = cv2.imread(pic_path)
    blue_image  = np.zeros(pic.shape)
    green_image = np.zeros(pic.shape)
    red_image   = np.zeros(pic.shape)
    blue_image[:, :, 0]  = pic[:, :, 0]
    green_image[:, :, 1] = pic[:, :, 1]
    red_image[:, :, 2]   = pic[:, :, 2]

    # Remove previous unicolor images
    file_list = os.listdir(channel_dir)
    for file_name in file_list:
        if (file_name.startswith("blue") or file_name.startswith("green") or file_name.startswith("red")):
            file_path = os.path.join(channel_dir, file_name)
            os.remove(file_path)

    # Save channels
    blue_path  = os.path.join(channel_dir, f"blue{id_pic}.jpg")
    green_path = os.path.join(channel_dir, f"green{id_pic}.jpg")
    red_path   = os.path.join(channel_dir, f"red{id_pic}.jpg")
    cv2.imwrite(blue_path , blue_image)
    cv2.imwrite(green_path, green_image)
    cv2.imwrite(red_path  , red_image)
