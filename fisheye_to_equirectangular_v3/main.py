# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from calibrate_camera import calibrate_camera
from import_camera_intrinsic_function import import_camera_intrinsic_function
from convert_image_to_equirectangular import convert_image_to_equirectangular
from glob import glob
import cv2

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # calibrate_camera(6,9, 22, glob('./CalibrationImages2/*.jpg'))
    image = cv2.imread("./FisheyePhotos/picture_sky.jpg")
    convert_image_to_equirectangular(image, image.shape[0], image.shape[1])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
