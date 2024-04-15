import os
import sys
import numpy as np
import cv2

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
checks_dir = os.path.join(data_dir, "checks/")

MAX_VAL = 255


# for the moment, basic intensity th
def get_uncertain_mask(pic, th):
    """
    Returns:
    Uncertain mask
    """
    pic_gray = cv2.cvtColor(pic.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, uncertain_mask = cv2.threshold(pic_gray, th, MAX_VAL, cv2.THRESH_BINARY)
    return uncertain_mask