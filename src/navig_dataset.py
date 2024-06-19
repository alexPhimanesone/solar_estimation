import os
import sys
import random
from utils import read_csv, read_all_csv

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir  = os.path.join(data_dir, "dataset/")
pics_dir     = os.path.join(dataset_dir, "pics/")
pprads_dir   = os.path.join(dataset_dir, "pprads/")
masks_dir    = os.path.join(dataset_dir, "masks/")
metadata_dir = os.path.join(dataset_dir, "metadata/")
checks_dir   = os.path.join(data_dir, "checks/")
zoom_dir     = os.path.join(checks_dir, "zoom/")
channel_dir  = os.path.join(checks_dir, "channel/")


def get_id_mask(id_pic=None):
    if id_pic is None:
        print("Provide id_pic")
        sys.exit(1)

    id_mask = read_csv(os.path.join(metadata_dir, "pics_metadata.csv"), id_pic, "id_mask")
    return id_mask


def get_id_endroit(id_pic=None, id_mask=None):
    if id_pic is None and id_mask is None:
        print("Provide id_pic or id_mask")
        sys.exit(1)
    
    if id_pic is None:
        data = read_all_csv(os.path.join(metadata_dir, "pics_metadata.csv"))
        for row in data:
            if row['id_mask'] == id_mask:
                id_pic = row['id_pic']
                break

    id_endroit = read_csv(os.path.join(metadata_dir, "pics_metadata.csv"), id_pic, "id_endroit")
    return id_endroit


def get_id_pprad(id_endroit=None, id_pic=None):
    if id_endroit is None and id_pic is None:
        print("Provide either id_endroit or id_pic")
        sys.exit(1)

    # Get id_endroit
    if id_endroit is None:
        id_endroit = read_csv(os.path.join(metadata_dir, "pics_metadata.csv"), id_pic, "id_endroit")
    
    # Get id_pprad from endroits_metadata.csv
    id_pprad = read_csv(os.path.join(metadata_dir, "endroits_metadata.csv"), id_endroit, "id_pprad")

    # If proper calibration
    if id_pprad != str(-1):
        return id_pprad
    
    # If no proper calibration
    else:
        # Get id_phone and id_lens
        id_phone = get_id_phone(id_endroit=id_endroit)
        id_lens = get_id_lens(id_endroit=id_endroit)

        # Choose a random corresponding pprad
        data = read_all_csv(os.path.join(metadata_dir, "pprads_metadata.csv"))
        corresponding_rows = [row for row in data if row['id_phone'] == id_phone and row['id_lens'] == id_lens]
        if corresponding_rows:
            random_row = random.choice(corresponding_rows)
            random_id_pprad = random_row['id_pprad']
            return random_id_pprad
        else:
            print("No pprad with this phone/lens couple.")    


def get_id_phone(id_pic=None, id_endroit=None):
    if id_endroit is None and id_pic is None:
        print("Provide either id_endroit or id_pic")
        sys.exit(1)

    if id_endroit is None:
        id_endroit = read_csv(os.path.join(metadata_dir, "endroits_metadata.csv"), id_pic, "id_endroit")
    id_phone = read_csv(os.path.join(metadata_dir, "endroits_metadata.csv"), id_endroit, "id_phone")
    return id_phone


def get_id_lens(id_pic=None, id_endroit=None):
    if id_endroit is None and id_pic is None:
        print("Provide either id_endroit or id_pic")
        sys.exit(1)

    if id_endroit is None:
        id_endroit = read_csv(os.path.join(metadata_dir, "endroits_metadata.csv"), id_pic, "id_endroit")
    id_lens = read_csv(os.path.join(metadata_dir, "endroits_metadata.csv"), id_endroit, "id_lens")
    return id_lens


def get_id_pic_list(id_endroit=None, id_mask=None):
    if not((id_endroit is None) ^ (id_mask is None)):
        print("Either use id_endroit or id_mask, not both.")
        sys.exit(1)
    data = read_all_csv(os.path.join(metadata_dir, "pics_metadata.csv"))
    id_pic_list = []
    
    if not(id_endroit is None):    
        for row in data:
            if row['id_endroit'] == str(id_endroit):
                id_pic_list.append(row['id_pic'])
    else:
        for row in data:
            if row['id_mask'] == str(id_mask):
                id_pic_list.append(row['id_pic'])
    
    print(f"len(id_pic_list): {len(id_pic_list)}")
    return id_pic_list
