import numpy as np
import csv


def read_raw_image(img_path, width=None, height=None, channel=3, dtype='uint8'):
    """
    Returns:
    Numpy array
    """

    with open(img_path, 'rb') as f:
        raw_data = f.read()
    image = np.frombuffer(raw_data, dtype=dtype)

    if width is None and height is None:
        print("Raw image resolution not provided. Trying to find it...")
        resolutions = [(4032, 3024), (4608, 3456)]
        for width, height in resolutions:
            try:
                
                image = image.reshape((height, width, channel))
                print("Found it")
                return image
            except ValueError:
                continue  # Move to the next resolution if reshaping fails
        raise ValueError("Please provide the resolution of the raw mask image.")
    else:
        image = image.reshape((height, width, channel))
        return image


def read_csv(csv_path, primary_id, field):
    column = []
    with open(csv_path, 'r') as f:
        csv_reader = csv.DictReader(f)    
        for row in csv_reader:
            column.append(row[field])
    return column[primary_id]
