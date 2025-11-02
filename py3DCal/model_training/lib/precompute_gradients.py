import os
import numpy as np
import pandas as pd
from PIL import Image

def precompute_gradients(dataset_path, annotation_path, r=36):
    """
    Computes the gradient map for a probe image. This is used to precompute the gradients for all images in the dataset for faster computation.

    Args:
        root_dir (str): The path of the data folder.
        csv_file (str): The name of the csv data file (must be located in 'root_dir').
        
    Returns:
        numpy.ndarray: A h x w x 2 numpy array with x and y gradient values for a circle located at the center.
    """
    # Read data file
    calibration_data = pd.read_csv(annotation_path)

    # Read the image
    image_path = os.path.join(dataset_path, "probe_images", calibration_data['img_name'][0])
    image = Image.open(image_path)
    image = np.asarray(image)

    # Get image height and width
    height, width, _ = image.shape

    # Get circle center and radius
    x = width // 2
    y = height // 2
    r = r
    
    # Create graident map
    gradient_map = np.zeros((height, width, 2))

    for i in range(height):
        for j in range(width):
            # Distance from pixel to center of circle
            d_center = np.sqrt((y - i) ** 2 + (x - j) ** 2)

            # If pixel is outside circle, set gradients to 0
            if d_center > r:
                Gx = 0
                Gy = 0
            
            # Otherwise, calculate the gradients
            else:
                normX = (j - x) / r
                normY = (i - y) / r
                normZ = np.sqrt(1 - normX ** 2 - normY ** 2)

                if normZ == 0:
                    normZ = 0.1
                
                Gx = normX / normZ
                Gy = normY / normZ

            # Update values in gradient map
            gradient_map[i,j] = np.array([Gx,Gy])

    return gradient_map