import os
import sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/ai/')))
from light_model import LightModel
from load import draw_id_pprad_list, PicsMasksDataset, ToTensor
from utils import get_str_date_time, print_logs

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir  = os.path.join(data_dir    , "dataset/")
pics_dir     = os.path.join(dataset_dir , "pics/")
masks_dir    = os.path.join(dataset_dir , "masks/")
training_dir = os.path.join(data_dir    , "Training/")


