import os
import sys
from utils import read_csv

data_dir = "Z:/Work/Projects/solar_estimation/data"
dataset_dir = os.path.join(data_dir, "dataset/")
metadata_dir = os.path.join(dataset_dir, "metadata/")


def get_id_endroit(id_pic=None):
    if id_pic is None:
        print("Provide id_pic")
        sys.exit(1)
    
    id_endroit = eval(read_csv(os.path.join(metadata_dir, "pics_metadata.csv"), id_pic, "id_endroit"))
    return id_endroit


def get_id_pprad(id_endroit=None, id_pic=None):
    if id_endroit is None and id_pic is None:
        print("Provide either id_endroit or id_pic")
        sys.exit(1)

    if id_endroit is None:
        id_endroit = eval(read_csv(os.path.join(metadata_dir, "pics_metadata.csv"), id_pic, "id_endroit"))
    id_pprad = eval(read_csv(os.path.join(metadata_dir, "endroits_metadata.csv"), id_endroit, "id_pprad"))
    return id_pprad
