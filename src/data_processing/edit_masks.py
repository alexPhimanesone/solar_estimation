import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import read_raw_image, write_raw_image, get_height_width , path_raw_to_jpg

data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
masking_dir = os.path.join(data_dir   , "masking")
dataset_dir = os.path.join(data_dir   , "dataset/")
pics_dir    = os.path.join(dataset_dir, "pics/")
masks_dir   = os.path.join(dataset_dir, "masks/")


def and_masks(mask1_path, mask2_path, mask_and_path, id_pic, invert1=False, invert2=False, write_im=False):
    
    # Load
    height, width = get_height_width(id_pic)
    mask1 = read_raw_image(mask1_path, width=width, height=height)
    mask2 = read_raw_image(mask2_path, width=width, height=height)

    # Compute mask_and
    true1 = 255 if invert1 else 0
    true2 = 255 if invert2 else 0
    mask_and = np.logical_and(mask1[:,:,0] == true1, mask2[:,:,0] == true2).astype(np.uint8) * 255

    # Save mask_and
    if write_im:
        cv2.imwrite(path_raw_to_jpg(mask1_path), mask1)
        cv2.imwrite(path_raw_to_jpg(mask2_path), mask2)
        cv2.imwrite(path_raw_to_jpg(mask_and_path), mask_and)
    write_raw_image(mask_and_path, mask_and)


def plot_line(id_pic, coo, axis):
    if not(axis == 'y' or axis == 'x'):
        print("axis is supposed to be 'y' or 'x'")
        sys.exit(1)
    axis_value = 0 if axis == 'y' else 1

    # Load
    pic_path = os.path.join(pics_dir, f"pic{id_pic}.jpg")
    pic = cv2.imread(pic_path)
    height, width = pic.shape[:2]
    if not(0 <= coo and coo <= pic.shape[axis_value]):
        print("coo is supposed to be between 0 and height/width")
        sys.exit(1)

    # Create pic_line
    if axis == 'y':
        pic_line = cv2.line(pic, (0, coo), (width - 1, coo) , (255, 0, 0), thickness=2)
    else:
        pic_line = cv2.line(pic, (coo, 0), (coo, height - 1), (255, 0, 0), thickness=2)

    # Show pic_line
    plt.figure()
    plt.imshow(pic_line)
    plt.show()

    

def join_masks(mask1_path, mask2_path, mask_join_path, id_pic, coo, axis, write_im=False):
    if not(axis == 'y' or axis == 'x'):
        print("axis is supposed to be 'y' or 'x'")
        sys.exit(1)
    axis_value = 0 if axis == 'y' else 1

    # Load
    pic_path = os.path.join(pics_dir, f"pic{id_pic}.jpg")
    pic = cv2.imread(pic_path)
    height, width = pic.shape[:2]
    mask1 = read_raw_image(mask1_path, width=width, height=height)
    mask2 = read_raw_image(mask2_path, width=width, height=height)
    if not(0 <= coo and coo <= pic.shape[axis_value]):
        print("coo is supposed to be between 0 and height/width")
        sys.exit(1)

    # Compute mask_join
    mask_join = np.zeros_like(pic)
    if axis == 'y':
        mask_join[:coo, :] = mask1[:coo, :]
        mask_join[coo:, :] = mask2[coo:, :]
    else:
        mask_join[:, :coo] = mask1[:, :coo]
        mask_join[:, coo:] = mask2[:, coo:]

    # Save mask_join
    if write_im:
        cv2.imwrite(path_raw_to_jpg(mask_join_path), mask_join)
    write_raw_image(mask_join_path, mask_join)


def plot_rectangle(id_mask, y_start, y_end, x_start, x_end, id_pic):
    
    # Load mask
    height, width = get_height_width(id_pic)
    mask_path = os.path.join(masks_dir, f"mask{id_mask}.raw")
    mask = read_raw_image(mask_path, width=width, height=height)
    mask_copy = mask.copy()

    # Create mask_rectangle
    mask_rectangle = cv2.rectangle(mask_copy, (x_start, y_start), (x_end, y_end), 128, 1)

    # Show mask_rectangle
    plt.figure()
    plt.imshow(mask_rectangle)
    plt.show()


def paint_mask(id_mask, y_start, y_end, x_start, x_end, color, id_pic):
    if not(color == 'black' or color == 'white'):
        print("axis is supposed to be 'black' or 'white'")
        sys.exit(1)
    color_value = 0 if color == 'black' else 255

    # Load mask
    height, width = get_height_width(id_pic)
    mask_path = os.path.join(masks_dir, f"mask{id_mask}.raw")
    mask = read_raw_image(mask_path, width=width, height=height)

    # Modify mask
    mask_painted = mask.copy()
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            mask_painted[y, x, :] = color_value
    
    # Save modified mask
    _, filename = os.path.split(mask_path)
    filename_painted = filename[:4] + "_painted" + filename[4:]
    mask_painted_path = os.path.join(masking_dir, filename_painted)
    write_raw_image(mask_painted_path, mask_painted)


def extend_black(mask_path, mask_extended_path, id_pic, radius=3,
                 y_start=None, y_end=None, x_start=None, x_end=None, write_im=False):
    height, width = get_height_width(id_pic)
    mask = read_raw_image(mask_path, width=width, height=height)
    mask_inverted = cv2.bitwise_not(mask)
    kernel = np.ones((2*radius+1, 2*radius+1), dtype=np.uint8)
    mask_inverted_extended = cv2.dilate(mask_inverted, kernel)
    mask_extended = cv2.bitwise_not(mask_inverted_extended)
    write_raw_image(mask_extended_path, mask_extended)
    if write_im:
        cv2.imwrite(path_raw_to_jpg(mask_extended_path), mask_extended)




'''
from fuse_masks import and_masks, join_masks

# Load pic
pic_path = os.path.join(pics_dir, f"pic{id_pic}.jpg")
pic = cv2.imread(pic_path)
height, width = pic.shape[:2]

# Load edges_th_outliers0009009
edges_th_outliers0009009_path = os.path.join(masking_dir, "edges_th_outliers0009009.raw")
edges_th_outliers0009009 = read_raw_image(edges_th_outliers0009009_path, width=width, height=height)

# Create white image
im_white = np.ones_like(pic) * 255
white_path = os.path.join(masking_dir, "white.raw")
write_raw_image(white_path, im_white)

# blank the bottom of edges_th_outliers0009009
coo = 1430
mask_join_path = os.path.join(masking_dir, "edges_white.raw")
join_masks(edges_th_outliers0009009_path, white_path, mask_join_path, id_pic, coo, axis='y')

# and_masks
current_mask_path = os.path.join(masks_dir, "mask0009066.raw")
result_path = os.path.join(masking_dir, "result.raw")
and_masks(mask_join_path, current_mask_path, result_path, id_pic, invert1=True, invert2=True)

# Write images
cv2.imwrite(path_raw_to_jpg(mask_join_path), read_raw_image(mask_join_path, width=width, height=height))
cv2.imwrite(path_raw_to_jpg(current_mask_path), read_raw_image(current_mask_path, width=width, height=height))
cv2.imwrite(path_raw_to_jpg(result_path), read_raw_image(result_path, width=width, height=height))
'''
