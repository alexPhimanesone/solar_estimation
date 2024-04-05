import os
import sys
import numpy as np
import cv2
import yaml as yml
import matplotlib.pyplot as plt
from data_processing.crop_around_disk import calib_pprad

data_dir = "C:/Users/aphimaneso/Work/Projects/solar_estimation/data/"
calibration_dir = os.path.join(data_dir, "calibration/")
calibs_dir = os.path.join(calibration_dir, "calibs/")
#pic_path = "C:/Users/aphimaneso/Work/Projects/solar_estimation/data/test/picture_sky.jpg"

calib_set_path = os.path.join(calibs_dir, "calib4/")
calib_pprad(calib_set_path)









