import sys
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir  = os.path.join(data_dir, "dataset/")
pics_dir     = os.path.join(dataset_dir, "pics/")
pprads_dir   = os.path.join(dataset_dir, "pprads/")
masks_dir    = os.path.join(dataset_dir, "masks/")
metadata_dir = os.path.join(dataset_dir, "metadata/")
checks_dir   = os.path.join(data_dir, "checks/")
zoom_dir     = os.path.join(checks_dir, "zoom/")
channel_dir  = os.path.join(checks_dir, "channel/")

blue  = (  0,   0, 255)
black = (  0,   0,   0)
red   = (255,   0,   0)
brown = (128,  64,   0)
white = (255, 255, 255)


def plot_pred(pic, pred, mask):
    height, width = pic.shape[:2]
    fig = np.zeros_like(pic)
    
    y_coords, x_coords = np.meshgrid(np.arange(0, height), np.arange(0, width))
    distances = (x_coords - np.floor(height/2))**2 + (y_coords - np.floor(width/2))**2
    disk = distances <= np.floor(height/2)**2

    TP = np.sum((disk == 1) & (pred == 255) & (mask == 255))  # TP
    TN = np.sum((disk == 1) & (pred ==   0) & (mask ==   0))  # TN
    FP = np.sum((disk == 1) & (pred == 255) & (mask ==   0))  # FP
    FN = np.sum((disk == 1) & (pred ==   0) & (mask == 255))  # FN
    nb_px_disk = np.product(pic.shape[:2]) * np.pi / 4
    print(f"TP: {TP / nb_px_disk}")
    print(f"TN: {TN / nb_px_disk}")
    print(f"FP: {FP / nb_px_disk}")
    print(f"FN: {FN / nb_px_disk}")

    fig = np.zeros_like(pic)
    fig[(disk == 1) & (pred == 255) & (mask == 255)] = blue
    fig[(disk == 1) & (pred ==   0) & (mask ==   0)] = black
    fig[(disk == 1) & (pred == 255) & (mask ==   0)] = red
    fig[(disk == 1) & (pred ==   0) & (mask == 255)] = brown
    fig[disk == 0] = white
    plt.imshow(fig)
    plt.show()
    