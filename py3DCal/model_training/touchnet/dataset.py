import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from ..lib.precompute_gradients import precompute_gradients
from ..lib.get_gradient_map import get_gradient_map


class TactileSensorDataset(Dataset):
    """
    Tactile Sensor dataset.
    """
    def __init__(self, dataset_path, annotation_path, blank_image_path, transform=None, radius=36):
        self.dataset_path = dataset_path
        self.annotation_path = annotation_path
        self.blank_image_path = blank_image_path
        self.transform = transform
        self.csv_file = dataset_path  # Store the CSV filename
        
        # Load the CSV data
        self.data = pd.read_csv(annotation_path, comment='#')
        
        self.precomputed_gradients = precompute_gradients(dataset_path, annotation_path, r=radius)
        self.blank_tensor = self.transform(Image.open(blank_image_path).convert("RGB"))
    
    def _add_coordinate_channels(self, image):
        """
        Add x and y coordinate channels to the input image.
        X channel: column indices (1s in first column, 2s in second column, etc.)
        Y channel: row indices (1s in first row, 2s in second row, etc.)
        """
        # Get image dimensions
        _, height, width = image.shape
        
        # Create x coordinate channel (column indices)
        x_coords = torch.arange(1, width + 1, dtype=torch.float32).unsqueeze(0).repeat(height, 1)
        x_channel = x_coords.unsqueeze(0)  # Add channel dimension
        
        # Create y coordinate channel (row indices)
        y_coords = torch.arange(1, height + 1, dtype=torch.float32).unsqueeze(1).repeat(1, width)
        y_channel = y_coords.unsqueeze(0)  # Add channel dimension
        
        # Concatenate original image with coordinate channels
        image_with_coords = torch.cat([image, x_channel, y_channel], dim=0)
        
        return image_with_coords
        
    def __len__(self):
        return len(self.data)  # Use the DataFrame length

    def __getitem__(self, idx):
        # Check if index is valid
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.dataset_path, self.data.iloc[idx, 0])
        img = Image.open(img_name).convert("RGB")
        image = np.array(img, copy=True)

        target = get_gradient_map(idx, annotation_path=self.annotation_path, precomputed_gradients=self.precomputed_gradients)

        # if self.transform:
        image = self.transform(image)
        target = self.transform(target)

        # Subtract pre-transformed blank tensor
        image = image - self.blank_tensor
        # Add coordinate channels
        image = self._add_coordinate_channels(image)

        sample = (image, target)
        return sample