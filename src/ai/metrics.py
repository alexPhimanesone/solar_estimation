import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# cv2 order is BGR
blue  = (255,   0,   0)
black = (  0,   0,   0)
red   = (  0,   0, 255)
brown = (  0,  64, 128)
white = (255, 255, 255)


def confusion_mat(pred, mask, disk_mask):

    # Compute confusion mat
    TP = np.sum((disk_mask == 1) & (pred ==   1) & (mask ==   1))
    TN = np.sum((disk_mask == 1) & (pred ==   0) & (mask ==   0))
    FP = np.sum((disk_mask == 1) & (pred ==   1) & (mask ==   0))
    FN = np.sum((disk_mask == 1) & (pred ==   0) & (mask ==   1))
    
    confusion_mat = {}
    confusion_mat['TP'] = TP
    confusion_mat['TN'] = TN
    confusion_mat['FP'] = FP
    confusion_mat['FN'] = FN
    return confusion_mat


def confusion_mat_rates(pred, mask, disk_mask):

    # Compute confusion mat
    TP = np.sum((disk_mask == 1) & (pred ==   1) & (mask ==   1))
    TN = np.sum((disk_mask == 1) & (pred ==   0) & (mask ==   0))
    FP = np.sum((disk_mask == 1) & (pred ==   1) & (mask ==   0))
    FN = np.sum((disk_mask == 1) & (pred ==   0) & (mask ==   1))
    nb_tot = TP + TN + FP + FN
    
    print(TP, TN, FP, FN, nb_tot)

    confusion_mat = {}
    confusion_mat['TP'] = TP / nb_tot
    confusion_mat['TN'] = TN / nb_tot
    confusion_mat['FP'] = FP / nb_tot
    confusion_mat['FN'] = FN / nb_tot
    return confusion_mat


def acc(pred, mask, disk_mask):
    cm = confusion_mat(pred, mask, disk_mask)
    nb_tot = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    nb_err = cm['FP'] + cm['FN']
    acc = 1 - nb_err / nb_tot
    return acc


def plot_cm(pred, mask, disk_mask, fig_path):
    height, width = pred.shape[:2]
    fig = np.zeros((height, width, 3), dtype=np.uint8)
    fig[(disk_mask == 1) & (pred ==   1) & (mask ==   1)] = blue
    fig[(disk_mask == 1) & (pred ==   0) & (mask ==   0)] = black
    fig[(disk_mask == 1) & (pred ==   1) & (mask ==   0)] = red
    fig[(disk_mask == 1) & (pred ==   0) & (mask ==   1)] = brown
    fig[disk_mask == 0] = 0
    cv2.imwrite(fig_path, fig)


def precision(pred, mask, disk_mask):
    TP = np.sum((disk_mask == 1) & (pred ==   1) & (mask ==   1))
    FP = np.sum((disk_mask == 1) & (pred ==   1) & (mask ==   0))
    precision = TP / (TP + FP)
    return precision


def recall(pred, mask, disk_mask):
    TP = np.sum((disk_mask == 1) & (pred ==   1) & (mask ==   1))
    FN = np.sum((disk_mask == 1) & (pred ==   0) & (mask ==   1))
    recall = TP / (TP + FN)
    return recall


def F1_score(pred, mask, disk_mask):
    precis = precision(pred, mask, disk_mask)
    rec = recall(pred, mask, disk_mask)
    f1_score = 2 * (1 / (1 / precis) + (1 / rec))
    return f1_score
