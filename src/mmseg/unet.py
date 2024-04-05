import os
import sys
import cv2
sys.path.append("C:/Users/aphimaneso/Work/Projects/solar_estimation/src/data_processing")
from crop_around_disk import crop_around_disk
from navig_dataset import get_id_pprad
from mmseg.apis import MMSegInferencer

data_dir = "Z:/Work/Projects/solar_estimation/data"
dataset_dir = os.path.join(data_dir, "dataset/")
pics_dir = os.path.join(dataset_dir, "pics/")
pprads_dir = os.path.join(dataset_dir, "pprads/")


# Prepare pic
id_pic = 0
pic_path = os.path.join(pics_dir, f"pic{id_pic}.jpg")
pic = cv2.imread(pic_path)
id_pprad = get_id_pprad(id_pic)
pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")
pic = crop_around_disk(pprad_path, pic)

# Run inference
model = 'unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024'
resolution = (512, 1024)
inferencer = MMSegInferencer(model=model)
inferencer(cv2.resize(pic, resolution), show=True, opacity=0.6)
