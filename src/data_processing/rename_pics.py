import os
import sys

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
dataset_dir = os.path.join(data_dir, "dataset/")
pics_dir = os.path.join(dataset_dir, "pics/")

nb_args = len(sys.argv) - 1
if not (nb_args == 2):
    print("Usage: python rename_pics.py <pp_phone> <ee_endroit>")
    sys.exit(1)
id_phone = sys.argv[1]
id_endroit = sys.argv[2]


# List all files in the directory
files = os.listdir(pics_dir)

# Iterate over each file
for filename in files:
    # Check if the file starts with "IMG_" and ends with ".jpg"
    if filename.startswith("IMG_") and filename.endswith(".jpg"):
        # Extract the numeric part of the filename
        numeric_part = filename.split("_")[1].split(".")[0]
        # Create the new filename
        new_filename = f"pic{id_phone}{id_endroit}"
        if len(numeric_part) == 1:
            new_filename += "00"
        if len(numeric_part) == 2:
            new_filename += "0"
        new_filename += f"{numeric_part}.jpg"
        # Rename the file
        if not(os.path.exists(os.path.join(pics_dir, new_filename))):
            os.rename(os.path.join(pics_dir, filename), os.path.join(pics_dir, new_filename))
