
import omnicalib as omni
import cv2
import numpy as np
import torch as torch

from colorama import Fore, Style
from import_camera_intrinsic_function import import_camera_intrinsic_function
from omnicalib.chessboard import get_points

def calibrate_camera(pattern_cols, pattern_rows, square_size, images, calibration_path, scatter_folder):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (pattern_cols, pattern_rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 2)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    ####### THIS SECTION IS FOR OCAMCALIB PORT #######
    object_points = omni.chessboard.get_points(pattern_rows, pattern_cols, float(square_size)).view(-1, 3)
    detections = {}

    # number of images that we managed to detect the checkerboard corners
    nb_images_used_for_calib = 0

    print(f"{Fore.YELLOW}Running through your calibration image set and detecting corners...{Style.RESET_ALL}")
    for fname in images:
        img_bw = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        # directly apply sharpening to original image (may lose some precision but that's lief ok)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_bw = cv2.filter2D(img_bw, -1, kernel)

        # downsize image (by 4)
        downsized_img_bw = cv2.resize(img_bw, (int(img_bw.shape[1]/4), int(img_bw.shape[0]/4)), interpolation=cv2.INTER_AREA)
        # sharpen it some more to make sure the corners stands out
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        downsized_img_bw = cv2.filter2D(downsized_img_bw, -1, kernel)

        # we find the checkboard pattern on the downsized image, then by knowing how stuffs remultiplies back up later
        ret, temp_corners = cv2.findChessboardCorners(downsized_img_bw, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == False:
            ret, temp_corners = cv2.findChessboardCornersSB(downsized_img_bw, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE)


        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)

            # remultiply the values (by 4 because corners detected on 4 time downsized image)
            corners = temp_corners * 4
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(img_bw, corners, (11, 11), (-1, -1), criteria)

            #
            imgpoints.append(corners2)

            ####### THIS SECTION IS FOR OCAMCALIB PORT #######
            detections[fname] = {'image_points': torch.from_numpy(corners2).to(torch.float64).squeeze(1),
                                 'object_points': object_points}

            nb_images_used_for_calib = nb_images_used_for_calib + 1
            print(f"{Fore.MAGENTA}Detected corners in " + fname + f"{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Unable to detect corners in " + fname + f"{Style.RESET_ALL}")

    print(f"{Fore.LIGHTMAGENTA_EX}Total number of images: " + str(len(images)) + f"{Style.RESET_ALL}")
    print(f"{Fore.LIGHTMAGENTA_EX}Total images with detected checkerboards: " + str(nb_images_used_for_calib) + f"{Style.RESET_ALL}")

    #assert nb_images_used_for_calib >= 8, "Below 8 images used for calibration, the result would not be precise"
    print(f"{Fore.GREEN}Done!{Style.RESET_ALL}")
    # run the calibration function

    print(f"{Fore.LIGHTYELLOW_EX}The calibration is started. Module courtesy of Thomas Pönitz, Github: https://github.com/tasptz/py-omnicalib{Style.RESET_ALL}")
    omni.main(detections, 4, 100, round(nb_images_used_for_calib / 4), scatter_folder, calibration_path)

    # run this here because we want it to write the estimated FOV into the calibration.yml, not really to get the camera data...
    #import_camera_intrinsic_function(calibration_path)

