import os
import sys
import numpy as np
sys.path.append(data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from utils import read_raw_image, write_raw_image

data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))
masking_dir   = os.path.join(data_dir     , "masking/")
dataset_dir   = os.path.join(data_dir     , "dataset/")
checks_dir    = os.path.join(data_dir     , "checks/")
pics_dir      = os.path.join(dataset_dir  , "pics/")
pprads_dir    = os.path.join(dataset_dir  , "pprads/")
masks_dir     = os.path.join(dataset_dir  , "masks/")
metadata_dir  = os.path.join(dataset_dir  , "metadata/")
zoom_dir      = os.path.join(checks_dir   , "zoom/")
channel_dir   = os.path.join(checks_dir   , "channel/")
inference_dir = os.path.join(data_dir     , "inference/")
in_dir        = os.path.join(inference_dir, "in/")


id_mask = "0009066"

# Load mask
mask_path = os.path.join(masks_dir, f"mask{id_mask}.raw")
mask = read_raw_image(mask_path)
print(mask.shape)

# Build mask_uni
mask_uni = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
mask_uni = np.all(mask == 255, axis=-1) * 255

# Print elements that don't match 255 or 0
non_255_0 = mask_uni == 0
non_255_0[mask_uni == 255] = False
non_255_0[mask_uni == 0] = False
print(mask[non_255_0])

# Format mask_uni
mask_uni = np.expand_dims(mask_uni, axis=-1)
print(mask_uni.shape)

# Write mask_uni
mask_uni_path = os.path.join(masks_dir, f"mask_uni{id_mask}.raw")
write_raw_image(mask_uni_path, mask_uni, dtype=np.uint8)
