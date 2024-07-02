import os
from os.path import join as opj
import sys
import numpy as np
import torch
import cv2
import csv
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai')))

data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
dataset_dir   = os.path.join(data_dir   , "dataset/")
pics_dir      = os.path.join(dataset_dir, "pics/")
pprads_dir    = os.path.join(dataset_dir, "pprads/")
masks_dir     = os.path.join(dataset_dir, "masks/")
metadata_dir  = os.path.join(dataset_dir, "metadata/")
preloaded_dir = os.path.join(dataset_dir, "preloaded/")
checks_dir    = os.path.join(data_dir   , "checks/")
zoom_dir      = os.path.join(checks_dir , "zoom/")
channel_dir   = os.path.join(checks_dir , "channel/")
training_dir  = os.path.join(data_dir   , "Training/")



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
    device = ("cuda" if torch.cuda.is_available()
     else "mps" if torch.backends.mps.is_available()
     else "cpu")
    print(f"Device: {device}")
    return device


def np_to_torch(arr, batched=None):
    if len(arr.shape) == 3 and batched is None:
        arr = np.transpose(arr, (2, 0, 1)) # (c, h, w)
    if len(arr.shape) == 4:
        arr = np.transpose(arr, (0, 3, 1, 2))
    tensor = torch.from_numpy(arr).to(torch.float32)
    return tensor


def torch_to_np(tensor, batched=None):
    arr = tensor.detach().numpy()
    if len(arr.shape) == 3 and batched is None:
        arr = np.transpose(arr, (1, 2, 0)) # (h, w, c)
    if len(arr.shape) == 4:
        arr = np.transpose(arr, (0, 2, 3, 1))
    return arr


def val_empty(root_dir):
    pics_dir = os.path.join(root_dir, "pics/")
    val_dir  = os.path.join(pics_dir, "val/")
    listval = os.listdir(val_dir)
    if listval is None or len(listval) == 0:
        return True
    else:
        return False


def get_model_path(str_date_time, epoch=None):
    save_dir = os.path.join(training_dir, str_date_time)
    if not(epoch is None):
        if isinstance(epoch, int):
            model_fn = f"model{epoch}.pt"
        if epoch == 'last':
            i = -1
            while os.path.exists(os.path.join(save_dir, f"model{i+1}.pt")): # ie "there's more after this one"
                i += 1
            model_fn = f"model{i}.pt"
    else:
        model_fn = "model.pt"
    model_path = os.path.join(save_dir, model_fn)
    return model_path


def get_last_model_path(train_timestamp):
    import re
    train_dir = opj(training_dir, train_timestamp)
    list_model_fn = [fn for fn in os.listdir(train_dir) if fn.endswith(".pth")]
    last_iter = 0
    for model_fn in list_model_fn:
        start = model_fn.find('_') + 1
        end = model_fn.find('.')
        iter = model_fn[start:end]
        if int(iter) > last_iter:
            last_iter = int(iter)
    last_model_fn = f"iter_{last_iter}.pth"
    last_model_path = opj(train_dir, last_model_fn)
    return last_model_path


def squeeze_mask(mask_3d):
    mask_2d = mask_3d[:, :, 0]
    mask_2d = mask_2d // 255
    return mask_2d


def get_vis_path(timestamp):
    save_dir = opj(training_dir, timestamp)
    print(save_dir)
    for root, _, files in os.walk(save_dir):
        if root.endswith("vis_data"):
            for file in files:
                if timestamp[:4] in file and timestamp[-4:] in file and file.endswith('.json'):
                    return os.path.join(root, file)
    return None
