import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io, transform
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/data_processing')))
from crop_around_disk import crop_around_disk, get_disk_mask_list
from navig_dataset import get_id_pprad, get_id_endroit, get_id_mask
from utils import read_all_csv, read_raw_image, np_to_torch

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir  = os.path.join(data_dir   , "dataset/")
pprads_dir   = os.path.join(dataset_dir, "pprads/")
metadata_dir = os.path.join(dataset_dir, "metadata/")
masks_dir    = os.path.join(dataset_dir, "masks/")


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
        nb_pics = len(self.pic_name_list)
        return nb_pics
    
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


def createPMDataloaders(datasets, batch_sizes):
    dls = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_sizes['train'], shuffle=True ),
             'val': torch.utils.data.DataLoader(datasets['val']  , batch_size=batch_sizes['val']  , shuffle=False),
            'test': torch.utils.data.DataLoader(datasets['test'] , batch_size=batch_sizes['test'] , shuffle=False)}
    return dls


'''
# to implement
class PreloadedPMDataset(Dataset):

    def __init__(self, dir, data_aug=None):
        self.dir            = dir
'''


# on implémentera la data aug plus tard

