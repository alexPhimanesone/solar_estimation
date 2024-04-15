import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/fisheye_to_equirectangular_v3')))
import numpy as np
import cv2
import yaml as yml
from glob import glob
from colorama import Fore, Style
import omnicalib as omni
from calibrate_camera import calibrate_camera
from camera_coords_to_image_intrinsic import camera_coords_to_image_intrinsic


def calibrate(calib_set_path):
    """
    Generates a calibration file.
    """

    # If not done for this calibration set, perform calibration (long computing)
    calibration_path = os.path.join(calib_set_path,
                                    os.path.basename(os.path.normpath(calib_set_path)) + ".yml")
    if not(os.path.exists(calibration_path)):
        print(f"{Fore.YELLOW}Calibrating...{Style.RESET_ALL}")

        data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
        calibration_dir = os.path.join(data_dir, "calibration/")
        config_dir = os.path.join(calibration_dir, "config/")
        plots_path = os.path.join(calib_set_path, "Plots/")

        with open(os.path.join(config_dir, "checkerboard.txt"), 'r') as f:
            checkerboard_info = f.readlines()
        pattern_cols = eval(checkerboard_info[0].strip())
        pattern_rows = eval(checkerboard_info[1].strip())
        square_size  = eval(checkerboard_info[2].strip())
        calib_im_list = glob(os.path.join(calib_set_path, "images/*.jpg"))
        calibrate_camera(pattern_cols, pattern_rows, square_size, calib_im_list, calibration_path, plots_path)


def estimate_radius(calib_set_path):
    """
    Save principal point and radius at pprad_path.
    """
    print(f"{Fore.YELLOW}Estimating FOV...{Style.RESET_ALL}")

    data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
    calibration_dir = os.path.join(data_dir, "calibration/")
    config_dir = os.path.join(calibration_dir, "config/")
    dataset_dir = os.path.join(data_dir, "dataset/")
    pprads_dir = os.path.join(dataset_dir, "pprads/")
    calibration_path = os.path.join(calib_set_path,
                                    os.path.basename(os.path.normpath(calib_set_path)) + ".yml")
    pprad_path = os.path.join(pprads_dir,
                              "pprad" + os.path.basename(os.path.normpath(calib_set_path))[len("calib"):] + ".yml")

    # PARAMETERS
    fov_steps_factor = 10

    with open(os.path.join(config_dir, "fov.txt"), 'r') as f:
        fov_limits = f.readlines()
    fov_min = eval(fov_limits[0].strip())
    fov_max = eval(fov_limits[1].strip())

    with open(calibration_path) as f:
        data = yml.load(f, Loader=yml.SafeLoader)
    poly_incident_angle_to_radius = data['poly_incident_angle_to_radius']
    principal_point = data['principal_point']

    # METHODE DE CAO
    fov_steps = (fov_max - fov_min) * fov_steps_factor
    fov_test_theta = np.linspace(fov_min, fov_max, fov_steps) * np.pi/180
    x_prime, y_prime  = np.tan(fov_test_theta), np.zeros(len(fov_test_theta))
    fov_limit = camera_coords_to_image_intrinsic(np.array([x_prime,y_prime]).T.tolist(),
                                                 poly_incident_angle_to_radius,
                                                 principal_point)
    index_of_max = np.argmax(np.transpose(fov_limit - principal_point))
    estimated_fov = fov_min + index_of_max / fov_steps_factor
    distance_to_fov = fov_limit[index_of_max][0] - principal_point[0]
    radius = round(distance_to_fov) + 1

    print(f"Estimated FOV: {estimated_fov}")
    data = {'principal_point': principal_point, 'radius': radius}
    with open(pprad_path, 'w') as f:
        yml.dump(data, f)


def get_disk(pprad):
    
    principal_point = pprad[0]
    radius = pprad[1]

    disk_points = []
    cx, cy = round(principal_point[0]), round(principal_point[1])
    for y in range(round(cy - radius) - 1, round(cy + radius) + 1):
        for x in range(round(cx - radius) - 1, round(cx + radius) + 1):
            if ((x - cx)**2 + (y - cy)**2 <= radius**2):
                disk_points.append((y, x))

    return disk_points


def crop_around_disk(pprad_path, img):
    """
    Returns:
    Minimal square image with black corners.
    """

    # Load pp and rad
    with open(pprad_path, 'r') as f:
        data = yml.load(f, Loader=yml.SafeLoader)
    principal_point = data['principal_point']
    radius = data['radius']

    # Calculate the bounding box for the disk region
    cx, cy = map(round, principal_point)
    x_min = cx - radius
    y_min = cy - radius
    x_max = cx + radius
    y_max = cy + radius

    # Crop the image using the bounding box
    cropped_img = img[y_min:y_max+1, x_min:x_max+1]

    # Calculate the coordinates of the disk points within the cropped image
    y_coords, x_coords = np.meshgrid(np.arange(y_min, y_max+1), np.arange(x_min, x_max+1))
    distances = (x_coords - cx)**2 + (y_coords - cy)**2
    mask = distances <= radius**2

    # Apply the mask to the cropped image
    cropped_img = np.where(mask[..., np.newaxis], cropped_img, 0)

    return cropped_img


def calib_pprad(calib_set_path):
    """
    Does the calibration, computes pprad, and plots results on an calibration image.
    """
    
    data_dir = "C:/Users/aphimaneso/Work/Projects/mmsegmentation/data/"
    dataset_dir = os.path.join(data_dir, "dataset/")
    pprads_dir = os.path.join(dataset_dir, "pprads/")
    pprad_path = os.path.join(pprads_dir,
                              "pprad" + os.path.basename(os.path.normpath(calib_set_path))[len("calib"):] + ".yml")
    plots_path = os.path.join(calib_set_path, "Plots/")

    calibrate(calib_set_path)
    estimate_radius(calib_set_path)

    # Select a random image from calib set images
    calib_im_list = glob(os.path.join(calib_set_path, "images/*.jpg"))
    pic_path = calib_im_list[0]
    image = cv2.imread(pic_path)

    # Plot img_disk
    img_disk = crop_around_disk(pprad_path, image)
    cv2.imwrite(os.path.join(plots_path, "img_disk.jpg"), img_disk)

    # Plot circle
    with open(pprad_path, 'r') as f:
        data = yml.load(f, Loader=yml.SafeLoader)
    cx, cy = data['principal_point']
    radius = data['radius']
    img_circle = cv2.circle(image, (round(cx), round(cy)), round(radius), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(plots_path, "img_circle.jpg"), img_circle)
