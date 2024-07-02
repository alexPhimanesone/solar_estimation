import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl
import json
from skimage import io, transform
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_processing')))
from crop_around_disk import crop_around_disk, get_disk_mask_list
from navig_dataset import get_id_pprad, get_id_endroit, get_id_mask
from utils import read_all_csv, read_raw_image, np_to_torch

data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))
dataset_dir   = os.path.join(data_dir   , "dataset/")
pprads_dir    = os.path.join(dataset_dir, "pprads/")
metadata_dir  = os.path.join(dataset_dir, "metadata/")
masks_dir     = os.path.join(dataset_dir, "masks/")
preloaded_dir = os.path.join(dataset_dir, "preloaded/")
training_dir  = os.path.join(data_dir   , "Training/")


def get_nb_endroit():
    data = read_all_csv(os.path.join(metadata_dir, "endroits_metadata.csv"))
    idx_max = 0
    for row in data:
        id_endroit = int(row['id_endroit'])
        if idx_max < id_endroit:
            idx_max = id_endroit
    nb_endroit = idx_max + 1
    return nb_endroit


def draw_id_pprad_list():
    nb_endroit = get_nb_endroit()
    id_pprad_list = np.zeros(nb_endroit, dtype=np.uint8)
    for e in range(nb_endroit):
        id_endroit = str(e)
        id_pprad_list[e] = get_id_pprad(id_endroit=id_endroit)
    return id_pprad_list


class PMDataset(Dataset):

    def __init__(self, dir, id_pprad_list, downsize_pic, downsize_mask, data_aug=None):
        self.dir            = dir
        self.pic_name_list  = os.listdir(self.dir)
        self.id_pprad_list  = id_pprad_list
        self.disk_mask_list = get_disk_mask_list(id_pprad_list)
        self.downsize_pic   = downsize_pic
        self.downsize_mask  = downsize_mask
        self.data_aug       = data_aug

        pic0 = self.__getitem__(0)['pic']
        self.resolution = pic0.size(dim=1), pic0.size(dim=2)
        
    def __len__(self):
        nb_disk_masks = len(self.pic_name_list)
        return pics_cat, masks_cat, disk_masks_cat
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Get id_pic and pprad
        id_pic = self.pic_name_list[idx][3:10]
        id_pprad = self.id_pprad_list[int(get_id_endroit(id_pic=id_pic))]
        pprad_path = os.path.join(pprads_dir, f"pprad{id_pprad}.yml")
        # Crop and downsize
        pic = io.imread(os.path.join(self.dir, f"pic{id_pic}.jpg"))
        pic = crop_around_disk(pprad_path, pic)
        pic = self.downsize_pic(np_to_torch(pic))
        mask = read_raw_image(os.path.join(masks_dir, f"mask{get_id_mask(id_pic=id_pic)}.raw"))
        mask = crop_around_disk(pprad_path, mask)
        mask[mask == 255] = 1
        mask = self.downsize_mask(np_to_torch(mask))
        # Apply data aug and format
        if self.data_aug:
            pic, mask = self.transform(pic, mask)
        return {'pic': pic, 'mask': mask}


def createPMDatasets(root_dir, id_pprad_list, downsize_pic, downsize_mask, data_aug=None):
    train_dir = os.path.join(root_dir, "train/")
    val_dir   = os.path.join(root_dir, "val/"  )
    test_dir  = os.path.join(root_dir, "test/" )
    datasets = {'train': PMDataset(train_dir, id_pprad_list, downsize_pic, downsize_mask, data_aug=data_aug),
                  'val': PMDataset(  val_dir, id_pprad_list, downsize_pic, downsize_mask),
                 'test': PMDataset( test_dir, id_pprad_list, downsize_pic, downsize_mask)}
    return datasets


class PreloadedPMDataset(Dataset):

    def __init__(self, rqm_subdir, data_aug=None):
        self.rqm_subdir    = rqm_subdir
        self.rqm_dir       = os.path.dirname(os.path.dirname(self.rqm_subdir))
        #                  = os.path.dirname(os.path.dirname(os.path.dirname(self.rqm_subdir)))
        self.pic_name_list = os.listdir(self.rqm_subdir)
        self.id_pprad_arr  = np.load(os.path.join(self.rqm_dir, "id_pprad_arr.npy"))
        self.disk_mask_arr = np.load(os.path.join(self.rqm_dir, "disk_mask_arr.npy"))
        with open(os.path.join(self.rqm_dir, "mask_dict.pkl"), 'rb') as f:
            self.mask_dict = pkl.load(f)
        self.data_aug      = data_aug
    
    def __len__(self):
        nb_pics = len(self.pic_name_list)
        return nb_pics
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Get id_pic
        id_pic = self.pic_name_list[idx][3:10]
        # Load and format
        pic       = io.imread(os.path.join(self.rqm_subdir, f"pic{id_pic}.png"))
        mask      = self.mask_dict[get_id_mask(id_pic=id_pic)]
        disk_mask = self.disk_mask_arr[int(get_id_endroit(id_pic))]
        pic       = np_to_torch(pic)
        mask      = np_to_torch(mask)
        disk_mask = np_to_torch(disk_mask)
        sample = {'pic': pic, 'mask': mask, 'disk_mask': disk_mask}
        # Apply data aug and format
        if self.data_aug:
            sample = self.transform(sample)
        return sample


def createPreloadedPMDatasets(rqm_dir, data_aug=None):
    train_dir = os.path.join(rqm_dir, "pics/", "train/")
    val_dir   = os.path.join(rqm_dir, "pics/", "val/"  )
    test_dir  = os.path.join(rqm_dir, "pics/", "test/" )
    datasets = {'train': PreloadedPMDataset(train_dir, data_aug=data_aug),
                  'val': PreloadedPMDataset(  val_dir),
                 'test': PreloadedPMDataset( test_dir)}
    return datasets


def createPMDataloaders(datasets, batch_sizes):
    dls = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_sizes['train'], shuffle=True ),
             'val': torch.utils.data.DataLoader(datasets['val']  , batch_size=batch_sizes['val']  , shuffle=False),
            'test': torch.utils.data.DataLoader(datasets['test'] , batch_size=batch_sizes['test'] , shuffle=False)}
    return dls


def get_dataloader(str_date_time, subdir):
    if not(subdir in ['train', 'val', 'test']):
        print("subdir must be 'train', 'val', or 'test'.")
        sys.exit(1)
    save_dir = os.path.join(training_dir, str_date_time)
    with open(os.path.join(save_dir, "hp.json"), 'r') as f:
        hp = json.load(f)
    rqm_dirname = f"{hp['resolution'][0]}x{hp['resolution'][1]}_qm{hp['qm']}"
    dataset = PreloadedPMDataset(os.path.join(preloaded_dir, rqm_dirname, "pics", subdir))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=hp['batch_sizes'][subdir], shuffle=False)
    return dataloader


def get_items_cat(str_date_time, subdir):
    dataloader = get_dataloader(str_date_time, subdir)
    items_list = {'pics': [], 'masks': [], 'disk_masks': []}
    for _, batch in enumerate(dataloader):
        pics, masks, disk_masks = batch['pic'], batch['mask'], batch['disk_mask']
        items_list['pics'      ].append(pics)
        items_list['masks'     ].append(masks)
        items_list['disk_masks'].append(disk_masks)
    pics_cat       = torch.cat(items_list['pics'      ], dim=0)
    masks_cat      = torch.cat(items_list['masks'     ], dim=0)
    disk_masks_cat = torch.cat(items_list['disk_masks'], dim=0)
    return pics_cat, masks_cat, disk_masks_cat





# on implémentera la data aug plus tard

