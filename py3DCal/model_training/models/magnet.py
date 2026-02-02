import os
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Union
from pathlib import Path

class MagNet(nn.Module):
    """
    MagNet: A PyTorch neural network for estimating contact location from ReSkin magnetometer readings.

    Args:
        load_pretrained (bool): If True, loads pretrained weights for the specified sensor type.
        root (str or pathlib.Path): The root directory for saving/loading the reskin_pretrained_weights.pth file.
    """
    def __init__(self, load_pretrained: bool = False, root: Union[str, Path] = "."):
        super().__init__()

        self._validate_parameters(load_pretrained, root)

        self.fc1 = nn.Linear(15, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 40)

        self.fc4 = nn.Linear(40, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 3)

        if load_pretrained:
            self._load_pretrained_model(root)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
    
        return x
    
    def _validate_parameters(self, load_pretrained, root):
        if load_pretrained and root is None:
            raise ValueError("root directory for storing/loading model cannot be None when load_pretrained is True.")
        
        if load_pretrained and not isinstance(root, (str, Path)):
            raise ValueError("root directory must be a valid file system path as a string or pathlib.Path object when load_pretrained is True.")

        if not load_pretrained and root != ".":
            print("Warning: root parameter is ignored when load_pretrained is False.")

    def _load_pretrained_model(self, root: Union[str, Path]):
        """
        Loads a pretrained model for the ReSkin sensor.
        Args:
            None.
        Returns:
            None.
        """

        file_path = os.path.join(root, "reskin_pretrained_weights.pth")

        # Check if ReSkin pretrained weights exist locally, if not download them
        if not os.path.exists(file_path):

            print(f"Downloading ReSkin pretrained weights ...")
            response = requests.get('https://zenodo.org/records/18462608/files/reskin_pretrained_weights.pth?download=1', stream=True)
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
            print(f"ReSkin pretrained weights already exists at: {file_path}/")

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