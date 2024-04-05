from yaml import load, SafeLoader
from numpy import tan, linspace, zeros, pi, array, argmax, transpose
import cv2
import sys
import os

from camera_coords_to_image_intrinsic import camera_coords_to_image_intrinsic
from colorama import Fore, Style

# this function import the data and estimate the fov of the camera
# note that if the FOV is above 89 it wouldnt know
# poly_incident_angle_to_radius is a numpy array
# principal point is a numpy array
# estimated_fov is the estimated FOV in DEGREES
def import_camera_intrinsic_function(calibration_path):
    # well principal point and the polynomial coeffs surely exist
    try:
        with open(calibration_path) as f:
            data = load(f, Loader=SafeLoader)
            poly_incident_angle_to_radius = data['poly_incident_angle_to_radius']
            principal_point = data['principal_point']
            estimated_fov = data['fov']
    except EnvironmentError:
        print(f'{Fore.LIGHTRED_EX}It seems that no camera calibration data was available. Please check for calibration.yml{Style.RESET_ALL}')
        sys.exit()
    except KeyError:
        f.close()
        os.remove(calibration_path)
        print(f'{Fore.CYAN}Just calibrated so we didnt estimate the FOV{Style.RESET_ALL}')
        ###################### ESTIMATE LENS FOV ######################
        print(f"{Fore.YELLOW}Estimating your lens FOV...{Style.RESET_ALL}")
        fov_test_theta = linspace(20,89,700)*pi/180    # probably wont have any thing worse than 20degs, hopefully ur not using one this bad for this work lol

        # directly estimate x_prime and y_prime (z_prime all =1) because this conversion is simple
        x_prime = tan(fov_test_theta)
        y_prime = zeros(len(fov_test_theta))

        # use the camera_coords_to_image_intrinsic to determine how all these points map onto the image
        fov_limit = camera_coords_to_image_intrinsic(array([x_prime,y_prime]).T.tolist(), poly_incident_angle_to_radius, principal_point)

        # the farthest point from the principal point is considered to be the limit
        # this is MY OPINION, I think that the poly_project_thetar would start to loop the angles back when it starts exceeding the FOV
        index_of_max = argmax(transpose(fov_limit - principal_point))
        print(index_of_max)
        estimated_fov = 20 + index_of_max/10

        data['fov'] = estimated_fov.tolist()
        with open(calibration_path, 'w') as f:
            import yaml
            yaml.dump(data, f, Dumper=yaml.SafeDumper)
            f.close()

        # we took one calibration image at random to draw the FOV circle as debug
        files = os.listdir('./CalibrationImages2')
        #image = cv2.imread("./CalibrationImages2/" + files[0])
        image = cv2.imread("FisheyePhotos/picture_sky.png")
        im_height, im_width, channels = image.shape

        distance_to_fov = fov_limit[index_of_max][0]-principal_point[0]

        # draw the FOV circle
        image = cv2.circle(image, (round(principal_point[0]),round(principal_point[1])), round(distance_to_fov), (0,0,255), 2)

        # write some FOV numbers in the 8 edges, they could not have possible made it so bad that it clip through all of these
        image = cv2.putText(image, str(estimated_fov), (round(principal_point[0] + distance_to_fov), round(principal_point[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        image = cv2.putText(image, str(estimated_fov), (round(principal_point[0] - distance_to_fov), round(principal_point[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        image = cv2.putText(image, str(estimated_fov), (round(principal_point[0]), round(principal_point[1] - distance_to_fov)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        image = cv2.putText(image, str(estimated_fov), (round(principal_point[0]), round(principal_point[1] + distance_to_fov)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        image = cv2.putText(image, str(estimated_fov),
                            (round(principal_point[0] + distance_to_fov/(2**(1/2))), round(principal_point[1] + distance_to_fov/(2**(1/2)))),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        image = cv2.putText(image, str(estimated_fov),
                            (round(principal_point[0] - distance_to_fov/(2**(1/2))), round(principal_point[1] + distance_to_fov/(2**(1/2)))),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        image = cv2.putText(image, str(estimated_fov),
                            (round(principal_point[0] + distance_to_fov/(2**(1/2))), round(principal_point[1] - distance_to_fov/(2**(1/2)))),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        image = cv2.putText(image, str(estimated_fov),
                            (round(principal_point[0] - distance_to_fov/(2**(1/2))), round(principal_point[1] - distance_to_fov/(2**(1/2)))),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imwrite('./DebugData/fov_test.jpg', image)
        print("ici")
        print(f"{Fore.GREEN}Estimated FOV is " + str(estimated_fov) + f"°{Style.RESET_ALL}")
        print(f"{Fore.LIGHTCYAN_EX}You should check to ensure that the FOV circle lines up with the border of your fisheye view! Image is saved to fov_test.jpg in DebugData if you need to see it again.{Style.RESET_ALL}")
        print(f"{Fore.LIGHTCYAN_EX}Close figure to continue...{Style.RESET_ALL}")
        cv2.imshow('fov_test',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    f.close()

    return poly_incident_angle_to_radius, principal_point, estimated_fov
