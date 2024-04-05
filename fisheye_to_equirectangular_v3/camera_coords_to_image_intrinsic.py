import torch

from omnicalib.projection import project_poly_thetar

# note that the camera_coords is [x_c, y_c]
def camera_coords_to_image_intrinsic (camera_coords, poly_incident_angle_to_radius, principal_point):
    # IMPORTANT: this function will not check for calibration.yml
    # so the main code has to ensure that it as calibration.yml
    # convert the camera

    camera_homo_coords = torch.Tensor(camera_coords)
    pad_z_coords = (0, 1)
    padded_camera_homo_coords = torch.nn.functional.pad(camera_homo_coords, pad_z_coords, "constant", 1.)
    image_coordinates = project_poly_thetar(padded_camera_homo_coords, poly_incident_angle_to_radius, torch.Tensor(principal_point), False).numpy().astype(int)

    return image_coordinates