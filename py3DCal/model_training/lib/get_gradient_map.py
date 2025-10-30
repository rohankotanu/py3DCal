import os
import numpy as np
import pandas as pd

def get_gradient_map(idx, annotation_path, precomputed_gradients):
    """
    Returns a gradient map for an image using precomputed gradients.
    Inputs:
        - idx: index of the image to use for gradient map
        - precomputed_gradients: precomputed gradients
        - root_dir: root directory of the dataset
        - csv_file: name of the csv file containing the sensor data
    """
    # Read data file
    sensor_data = pd.read_csv(annotation_path, comment='#')

    height, width, _ = precomputed_gradients.shape

    x = int(float(sensor_data['x_px'][idx]))
    y = int(float(sensor_data['y_px'][idx]))

    right_shift = x - width // 2
    down_shift = y - height // 2

    offset = max(abs(right_shift), abs(down_shift))

    gradient_map = np.zeros((height + offset * 2, width + offset * 2, 2))
    gradient_map[:,:,0] = np.pad(precomputed_gradients[:,:,0], pad_width=offset, mode='constant')
    gradient_map[:,:,1] = np.pad(precomputed_gradients[:,:,1], pad_width=offset, mode='constant')

    # Shift the array 1 position to the right along the horizontal axis (axis=1)
    gradient_map = np.roll(gradient_map, right_shift, axis=1)

    # Shift the array 1 position down along the vertical axis (axis=0)
    gradient_map = np.roll(gradient_map, down_shift, axis=0)

    gradient_map = gradient_map[offset:offset+height, offset:offset+width]

    return gradient_map
