import os
import sys
import numpy as np
import cv2
import csv

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir  = os.path.join(data_dir, "dataset/")
pics_dir     = os.path.join(dataset_dir, "pics/")
pprads_dir   = os.path.join(dataset_dir, "pprads/")
masks_dir    = os.path.join(dataset_dir, "masks/")
metadata_dir = os.path.join(dataset_dir, "metadata/")
checks_dir   = os.path.join(data_dir, "checks/")
zoom_dir     = os.path.join(checks_dir, "zoom/")
channel_dir  = os.path.join(checks_dir, "channel/")


def read_raw_image(img_path, width=None, height=None, dtype='uint8'):
    """
    Returns:
    Numpy array
    """
    from itertools import product
    channels = [3, 1]

    with open(img_path, 'rb') as f:
        raw_data = f.read()
    image = np.frombuffer(raw_data, dtype=dtype)

    if width is None or height is None:
        resolutions = [(3024, 4032), (3456, 4608)]
        shapes = [(*resolution, channel) for resolution, channel in product(resolutions, channels)]
        for shape in shapes:
            try:
                image = image.reshape(shape)
                return image
            except ValueError:
                continue  # Move to the next shape if reshaping fails
        print(f"data shape: {image.shape}")
        raise ValueError("Please provide the resolution of the raw mask image.")
    
    else:
        for channel in channels:
            try:
                image = image.reshape((height, width, channel))
                return image
            except ValueError:
                continue # Move to the next channel if reshaping fails
        print(f"data shape: {image.shape}")
        raise ValueError("Reshaping failed")


def read_csv(csv_path, primary_id, field_name):
    with open(csv_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            primary_key = list(row.keys())[0]
            if row[primary_key] == primary_id:
                return row[field_name]
    print("Invalid primary_id")
    print(primary_id)
    sys.exit(1)


def read_all_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data


def write_raw_image(img_path, img, dtype=np.uint8):
    img_c = np.ravel(img, order='C').astype(dtype)
    with open(img_path, 'wb') as f:
        f.write(img_c)


def path_raw_to_jpg(img_path_raw):
    folder, filename = os.path.split(img_path_raw)
    base_name, _ = os.path.splitext(filename)
    img_path_jpg = os.path.join(folder, base_name + ".jpg")
    return img_path_jpg


def get_height_width(id_pic):
    pic_path = os.path.join(pics_dir, f"pic{id_pic}.jpg")
    pic = cv2.imread(pic_path)
    return pic.shape[:2]


def remove_first_zero(string):
    if string[0] == '0':
        string = string[1:]
    return string


def mult_channels(im_single_channel):
    height, width = im_single_channel.shape[:2]
    print(im_single_channel.shape)
    im_mult_channels = np.zeros((height, width, 3))
    im_mult_channels[im_single_channel[:, :, 0] ==   0] = (  0,   0,   0)
    im_mult_channels[im_single_channel[:, :, 0] == 255] = (255, 255, 255)
    return im_mult_channels


def get_str_date_time():
    import calendar
    import time
    from datetime import datetime

    current_GMT = time.gmtime()
    timestamp = calendar.timegm(current_GMT)
    date_time = datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%m%d-%H%M")
    print(f"Current timestamp: {str_date_time}")
    return str_date_time


def print_logs(epoch, train_loss, val_loss=None, stop_wait_time=None):
    print(f"Epoch {epoch}")
    print(f"Train loss: {train_loss:.4f}")
    if val_loss:
        print(f"Val   loss: {val_loss:.4f}")
    if stop_wait_time:
        print(f"Stop_wait_time: {stop_wait_time}")


def write_hp(save_dir, hp):
    import json
    hp_json = json.dumps(hp)
    with open(os.path.join(save_dir, "hp.json"), 'w') as f:
        f.write(hp_json)


def get_device():
    import torch
    device = ("cuda" if torch.cuda.is_available()
     else "mps" if torch.backends.mps.is_available()
     else "cpu")
    print(f"Device: {device}")
    return device


def str_to_loss(loss_name):
    import torch
    loss = None
    if loss_name == 'BCEWithLogits':
        loss = torch.nn.BCEWithLogitsLoss()
    return loss


def np_to_torch(im):
    import torch
    return torch.from_numpy(np.transpose(im, (2, 0, 1))).to(torch.float32)


def val_empty(root_dir):
    pics_dir = os.path.join(root_dir, "pics/")
    val_dir  = os.path.join(pics_dir, "val/")
    if os.listdir(val_dir) is None:
        return True
    else:
        return False
