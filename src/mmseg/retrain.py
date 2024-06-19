
import os
from os.path import join as opj
import sys
'''
import torch
import torchvision
import mmseg
import mmcv
import mmengine
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches
'''
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import Config
from mmengine.runner import Runner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/')))
from utils import get_str_date_time

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data"
dataset_dir = opj(data_dir, "dataset")
training_dir = opj(data_dir, "Training/")
save_dir = opj(training_dir, get_str_date_time())
data_root = opj(dataset_dir, "mmseg_orga/", "cropped")

classes = ('sky', 'nonsky')
palette = [[0, 0, 0], [255, 255, 255]]


@DATASETS.register_module()
class SkyDetectionDataset(BaseSegDataset):
    METAINFO = dict(classes = classes, palette = palette)
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)


def get_cfg():
    
    cfg = Config.fromfile('mmseg/configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py')

    cfg.norm_cfg = dict(requires_grad=True, type='BN') # Since we use only one GPU, BN is used instead of SyncBN
    cfg.crop_size = (256, 256)
    cfg.model.data_preprocessor.size = cfg.crop_size
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 2
    cfg.model.auxiliary_head.num_classes = 2
    cfg.model.decode_head.out_channels = 2
    cfg.model.auxiliary_head.out_channels = 2
    cfg.model.decode_head.loss_decode.use_sigmoid = False
    cfg.model.auxiliary_head.loss_decode.use_sigmoid = False

    # Modify dataset type and path
    cfg.dataset_type = 'SkyDetectionDataset'
    cfg.data_root = data_root

    cfg.train_dataloader.batch_size = 8

    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
        dict(type='RandomRotate', prob=1, degree=180, pad_val=0, seg_pad_val=0),
        dict(type='RandomFlip', prob=0.5),
        #dict(type='PhotoMetricDistortion'),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=1.0), # default cat_max_ratio is 0.75
        dict(type='PackSegInputs')
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(512, 512), keep_ratio=True),
        # add loading annotation after ``Resize`` because ground truth
        # does not need to do resize data transform
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    ]

    cfg.model.test_cfg = dict(crop_size=(256, 256), mode='slide', stride=(170, 170))

    cfg.train_dataloader.dataset.type = cfg.dataset_type
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix = dict(img_path=opj(data_root, "img_dir", "train"),
                                                    seg_map_path=opj(data_root, "ann_dir", "train"))
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline

    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root 
    cfg.val_dataloader.dataset.data_prefix = dict(img_path=opj(data_root, "img_dir", "test"),
                                                seg_map_path=opj(data_root, "ann_dir", "test"))
    cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline

    cfg.test_dataloader = cfg.val_dataloader

    # Load the pretrained weights
    cfg.load_from = "mmseg/configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth"

    # Set up working dir to save files and logs.
    cfg.work_dir = save_dir

    cfg.train_cfg.max_iters = 250
    cfg.train_cfg.val_interval = 250
    cfg.default_hooks.logger.interval = 25
    cfg.default_hooks.checkpoint.interval = 50

    # Set seed to facilitate reproducing the result
    cfg['randomness'] = dict(seed=0)

    return cfg


if __name__ == '__main__':
    cfg = get_cfg()
    print('Training is gonna start!')
    runner = Runner.from_cfg(cfg)
    runner.train()
