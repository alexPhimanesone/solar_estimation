import os
import sys
import numpy as np
import cv2
import csv
from itertools import product

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
    channels = [3, 1]

    with open(img_path, 'rb') as f:
        raw_data = f.read()
    image = np.frombuffer(raw_data, dtype=dtype)

    if width is None or height is None:
        print("Raw image resolution not provided. Trying to find it...")
        resolutions = [(3024, 4032), (3456, 4608)]
        shapes = [(*resolution, channel) for resolution, channel in product(resolutions, channels)]
        for shape in shapes:
            try:
                image = image.reshape(shape)
                print("Found it")
                return image
            except ValueError:
                continue  # Move to the next shape if reshaping fails
        raise ValueError("Please provide the resolution of the raw mask image.")
    
    else:
        for channel in channels:
            try:
                image = image.reshape((height, width, channel))
                return image
            except ValueError:
                continue # Move to the next channel if reshaping fails
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


def write_raw_image(img_path, img):
    img_c = np.ravel(img, order='C')
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
