import os
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Union
from pathlib import Path

class SensorType(Enum):
    """
    SensorType: Available sensor types with pretrained weights and compiled datasets
    """
    DIGIT = "DIGIT"
    GELSIGHTMINI = "GelSightMini"

class TouchNet(nn.Module):
    """
    TouchNet: A PyTorch neural network for producing surface normal maps from tactile sensor images.

    Args:
        load_pretrained (bool): If True, loads pretrained weights for the specified sensor type.
        sensor_type (SensorType): The type of tactile sensor. Must be specified if load_pretrained is True.
        root (str or pathlib.Path): The root directory for saving/loading the pretrained_weights (.pth) file.
    """
    def __init__(self, load_pretrained: bool = False, sensor_type: SensorType = None, root: Union[str, Path] = "."):
        super().__init__()

        self._validate_parameters(load_pretrained, sensor_type, root)

        self.conv1 = nn.Conv2d(5, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(0.3)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.dropout5 = nn.Dropout2d(0.3)

        self.conv6 = nn.Conv2d(256, 128, kernel_size=5, padding=2)
        self.bn6 = nn.BatchNorm2d(128)
        self.dropout6 = nn.Dropout2d(0.2)

        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)   
        self.dropout7 = nn.Dropout2d(0.2)

        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.dropout8 = nn.Dropout2d(0.2)

        self.conv9 = nn.Conv2d(32, 2, kernel_size=1)

        if load_pretrained:
            self._load_pretrained_model(root, sensor_type)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout5(x)

        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout6(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout7(x)

        x = F.relu(self.bn8(self.conv8(x)))
        x = self.dropout8(x)

        x = self.conv9(x)

        return x
    
    def _validate_parameters(self, load_pretrained, sensor_type, root):
        if load_pretrained and sensor_type is None:
            raise ValueError("sensor_type must be specified when load_pretrained is True. sensor_type must be either SensorType.DIGIT or SensorType.GELSIGHTMINI.")

        if load_pretrained and not isinstance(sensor_type, SensorType):
            raise ValueError("sensor_type must be either SensorType.DIGIT or SensorType.GELSIGHTMINI.")
        
        if load_pretrained and root is None:
            raise ValueError("root directory for storing/loading model cannot be None when load_pretrained is True.")
        
        if load_pretrained and not isinstance(root, (str, Path)):
            raise ValueError("root directory must be a valid file system path as a string or pathlib.Path object when load_pretrained is True.")

        if not load_pretrained and sensor_type is not None:
            print("Warning: sensor_type parameter is ignored when load_pretrained is False.")

        if not load_pretrained and root is not ".":
            print("Warning: root parameter is ignored when load_pretrained is False.")

    def _load_pretrained_model(self, root, sensor_type):
        """
        Loads a pretrained model for either the DIGIT or GelSightMini sensor.
        Args:
            None.
        Returns:
            None.
        """
        
        if sensor_type == SensorType.DIGIT:
            file_path = os.path.join(root, "digit_pretrained_weights.pth")

            # Check if DIGIT pretrained weights exist locally, if not download them
            if not os.path.exists(file_path):

                print(f"Downloading DIGIT pretrained weights ...")
                response = requests.get('https://zenodo.org/records/17517028/files/digit_pretrained_weights.pth?download=1', stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024

                # Save file in chunks to handle large datasets
                with open(file_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc="Downloading",
                    ncols=80
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                print(f"Download complete!")
            else:
                print(f"DIGIT pretrained weights already exists at: {file_path}/")

        elif sensor_type == SensorType.GELSIGHTMINI:
            file_path = os.path.join(root, "gsmini_pretrained_weights.pth")

            # Check if GelSight Mini pretrained weights exist locally, if not download them
            if not os.path.exists(file_path):

                print(f"Downloading GelSight Mini pretrained weights ...")
                response = requests.get('https://zenodo.org/records/17517028/files/gsmini_pretrained_weights.pth?download=1', stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024

                # Save file in chunks to handle large datasets
                with open(file_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc="Downloading",
                    ncols=80
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                print(f"Download complete!")
            else:
                print(f"GelSight Mini pretrained weights already exists at: {file_path}/")

        state_dict = torch.load(file_path, map_location="cpu")

        self.load_state_dict(state_dict)

    def load_weights(self, weights_path: Union[str, Path]):
        """
        Loads model weights from a specified .pth file.

        Args:
            weights_path (str or pathlib.Path): The file path to the .pth file containing the model weights.
        Returns:
            None.
        Raises:
            ValueError: If the weights_path is not specified or invalid.
        """
        if weights_path is None:
            raise ValueError("weights_path cannot be None.")

        if not isinstance(weights_path, (str, Path)):
            raise ValueError("weights_path must be a valid file system path as a string or pathlib.Path object.")

        if not os.path.exists(weights_path):
            raise ValueError(f"The specified weights_path does not exist: {weights_path}")

        state_dict = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(state_dict)