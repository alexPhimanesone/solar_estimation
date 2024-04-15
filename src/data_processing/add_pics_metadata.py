import os
import sys
import csv

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir = os.path.join(data_dir, "dataset/")
metadata_dir = os.path.join(dataset_dir, "metadata/")
pics_dir = os.path.join(dataset_dir, "pics/")

csv_path = os.path.join(metadata_dir, "pics_metadata.csv")
fields = ["id_pic", "id_mask", "id_endroit"]

nb_args = len(sys.argv) - 1
if not (nb_args == 3):
    print("Usage: python rename_pics.py <pp_phone> <ee_endroit> <id_mask>")
    sys.exit(1)
pp_phone = sys.argv[1]
ee_endroit = sys.argv[2]
id_mask = sys.argv[3]


# Get id_pic list
matching_pic_paths = [file for file in os.listdir(pics_dir) if file[3:7] == pp_phone + ee_endroit]
matching_pic_paths = sorted(matching_pic_paths)

# Format data to write
data = []
for pic_path in matching_pic_paths:
    row = {"id_pic": pic_path[3:10], "id_mask": id_mask, "id_endroit": eval(ee_endroit.lstrip('0'))}
    data.append(row)

# Add rows to csv
with open(csv_path, mode='a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    for row in data:
        writer.writerow(row)
