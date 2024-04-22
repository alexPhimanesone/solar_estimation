import sys
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir  = os.path.join(data_dir   , "dataset/")
pics_dir     = os.path.join(dataset_dir, "pics/")
pprads_dir   = os.path.join(dataset_dir, "pprads/")
masks_dir    = os.path.join(dataset_dir, "masks/")
metadata_dir = os.path.join(dataset_dir, "metadata/")
checks_dir   = os.path.join(data_dir   , "checks/")
zoom_dir     = os.path.join(checks_dir , "zoom/")
channel_dir  = os.path.join(checks_dir , "channel/")


def divide_into_parts(a, b):
    quotient = a // b
    remainder = a % b
    parts = [quotient] * b
    for i in range(remainder):
        parts[i] += 1
    return parts


def quantile_pooling(image, resolution, disk_mask, q=1/2):

    height, width = image.shape[:2]
    output_height, output_width = resolution
    kernel_list_y, kernel_list_x = divide_into_parts(height, output_height), divide_into_parts(width, output_width)

    output = np.zeros(resolution, dtype=np.uint8)
    output_disk_mask = np.zeros(resolution)
    y_end = 0
    for y in range(output_height):
        y_start = y_end
        y_end = y_start + kernel_list_y[y]
        x_end = 0
        for x in range(output_width):
            x_start = x_end
            x_end = x_start + kernel_list_x[x]

            # Take the minimum value from the region
            region = image[y_start:y_end, x_start:x_end][disk_mask[y_start:y_end, x_start:x_end]]
            if region.size > 0:
                value = np.quantile(region, q)
                output_disk_mask[y, x] = 1
            else:
                value = 0
            output[y, x] = value

    return output, output_disk_mask
