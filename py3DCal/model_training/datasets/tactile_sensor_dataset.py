import os
import torch
import pandas as pd
from PIL import Image
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from ..lib.precompute_gradients import precompute_gradients
from ..lib.get_gradient_map import get_gradient_map
from ..lib.add_coordinate_embeddings import add_coordinate_embeddings
from ..lib.validate_parameters import validate_root

class TactileSensorDataset(Dataset):
    """
    Tactile Sensor Dataset.

    Args:
        root (str or pathlib.Path): The root directory that contains the dataset folder.
        add_coordinate_embeddings (bool, optional): If True, adds xy coordinate embeddings to each image. Defaults to True.
        subtract_blank (bool, optional): If True, subtracts a blank image from each input image. Defaults to False.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: ``transforms.ToTensor()``
    """
    def __init__(self, root: Union[str, Path], add_coordinate_embeddings=True, subtract_blank=True, transform=None):
        validate_root(root)

        self.root = root
        self.annotation_path = os.path.join(root, "annotations", "annotations.csv")
        self.metadata_path = os.path.join(root, "annotations", "metadata.json")
        self.blank_image_path = os.path.join(root, "blank_images", "blank.png")
        self.add_coordinate_embeddings = add_coordinate_embeddings
        self.subtract_blank = subtract_blank

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        # Load the CSV data
        self.data = pd.read_csv(self.annotation_path)

        # Get probe radius (in px) from metadata
        metadata = pd.read_json(self.metadata_path, typ="series")
        radius = metadata["probe_radius_mm"] * metadata["px_per_mm"]

        # Precompute gradients
        self.precomputed_gradients = precompute_gradients(dataset_path=self.root, annotation_path=self.annotation_path, r=radius)

        # Load and transform blank image
        if subtract_blank:
            self.blank_image = self.transform(Image.open(self.blank_image_path).convert("RGB"))

    def __len__(self):
        return len(self.data)  # Use the DataFrame length

    def __getitem__(self, idx):
        # Check if index is valid
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = os.path.join(self.root, "probe_images", self.data.iloc[idx, 0])
        image = Image.open(image_name).convert("RGB")

        target = get_gradient_map(idx, annotation_path=self.annotation_path, precomputed_gradients=self.precomputed_gradients)

        image = self.transform(image)
        target = self.transform(target)

        if self.subtract_blank:
            # Subtract pre-transformed blank tensor
            image = image - self.blank_image

        if self.add_coordinate_embeddings:
            # Add coordinate embeddings
            image = add_coordinate_embeddings(image)

        sample = (image, target)

        return sample