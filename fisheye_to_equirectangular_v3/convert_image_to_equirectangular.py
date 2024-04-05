import numpy as np
import cv2
from camera_coords_to_image_intrinsic import camera_coords_to_image_intrinsic
from import_camera_intrinsic_function import import_camera_intrinsic_function
from colorama import Fore, Style

def convert_image_to_equirectangular(image, im_height, im_width):
    print(f"{Fore.YELLOW}Performing remapping...{Style.RESET_ALL}")

    poly_incident_angle_to_radius, principal_point, estimated_fov = import_camera_intrinsic_function()

    azimuth_length = 1920
    zenith_length = 480
    estimated_fov = estimated_fov*np.pi/180     # convert to RADIANS

    # construct the array of index to iterate thru azimuth + zenith and the respective blank image
    horizontal_index = np.linspace(0, azimuth_length-1, azimuth_length, True, False, 'int')
    vertical_index = np.linspace(0, zenith_length-1, zenith_length, True, False, 'int')

    # each point on the conformal image has an equivalent azimuth and zenith
    azimuth_remapped = horizontal_index*2*np.pi/azimuth_length
    zenith_remapped = vertical_index*estimated_fov/zenith_length

    # create a meshgrid of coordinates. Note that the zenith section is in one mat zenith_mat and azimuth is in the other mat
    zenith_mat, azimuth_mat = np.meshgrid(zenith_remapped, azimuth_remapped, indexing='ij')

    # multiply those two mat to create the max of x_prime and y_prime
    x_prime = np.multiply(np.cos(azimuth_mat), np.tan(zenith_mat))
    y_prime = np.multiply(np.sin(azimuth_mat), np.tan(zenith_mat))

    # stack x_prime and y_prime to create a matrix of coordinate pairs. This could be used directly with intrinsic.camera_coords_to_image_intrinsic
    xy_coord_matrix = np.stack((x_prime,y_prime),axis=2)

    # this contains all the image coords of the remapped points
    equi_point = camera_coords_to_image_intrinsic(xy_coord_matrix.tolist(), poly_incident_angle_to_radius, principal_point)

    # create a blank canvas utilizing the azimuth_length and zenith_length values
    # the dtype=np.unit8 is important to make the imshow function understand how to plot the image.
    # this section is a bit confusing... OpenCV/Omnicalib convention is x to the right, y to the bottom, z into image
    # while accessing the image manually would have the first argument toward the bottom and second argument to the right
    # this means that the first component in the equi_point is actually the HORIZONTAL coordinates and the second component is the VERTICAL COORDINATES
    # so you need to check the [0] component with im_width and [1] with im_height
    # then the access is done as image[equi_point[ver][hor][1]][equi_point[ver][hor][0]]
    conformal_image = np.zeros((zenith_length, azimuth_length, 3), dtype=np.uint8)
    conversion_info_image = np.zeros((im_height, im_width), dtype=np.uint8)

    for ver in vertical_index:
        for hor in horizontal_index:
            if equi_point[ver][hor][0] >= 0 and equi_point[ver][hor][0] < im_width and equi_point[ver][hor][1] >= 0 and equi_point[ver][hor][1] < im_height:
                    conformal_image[ver][hor] = image[equi_point[ver][hor][1]][equi_point[ver][hor][0]]
                    conversion_info_image[equi_point[ver][hor][1]][equi_point[ver][hor][0]] = conversion_info_image[equi_point[ver][hor][1]][equi_point[ver][hor][0]] + 1

    
    height = 3024
    width = 4032
    for i in range(500, height - 500):
         if (i%500 == 0):
              print(i)
         for j in range(700, width - 700):
            val = conversion_info_image[i, j]
            if val == 0:
                 val_new = 0
            elif val == 1:
                 val_new = 1
            elif val >= 20:
                 val_new = 4
            elif val > 5:
                 val_new = 3
            conversion_info_image[i, j] = val_new
    import matplotlib.pyplot as plt
    plt.imshow(conversion_info_image, cmap='viridis')
    plt.colorbar()
    plt.show()
    


    cv2.imwrite('./remapped.jpg', conformal_image)
    cv2.imwrite('./remapping_coverage.jpg',(conversion_info_image.astype(float)*255/np.max(conversion_info_image)).astype(int))
    print(np.max(conversion_info_image))
    print(f'{Fore.GREEN}Finished equirectangular remapping' + f'{Style.RESET_ALL}')
