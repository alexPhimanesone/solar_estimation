import os
import sys
import cv2
sys.path.append("C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/data_processing")
sys.path.append("C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/mmseg/configs")
from crop_around_disk import load_and_crop_pic
from mmseg.apis import init_model, MMSegInferencer

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
checkpoints_dir = os.path.join(data_dir, "checkpoints/")
configs_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/mmseg/configs/"

id_pic = "0010002"
'''
config_path = os.path.join(configs_dir, 'unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py')
checkpoint_path = os.path.join(checkpoints_dir,
                               "fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth")
'''
resolution = (512, 512)

def inference():
    # Run inference
    pic = load_and_crop_pic(id_pic)
    #model = init_model(config_path, checkpoint_path, 'cpu')
    model = 'unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024'
    inferencer = MMSegInferencer(model=model)
    result = inferencer(cv2.resize(pic, resolution), show=True, opacity=0.6)
    pred = result['predictions']
    return pred

