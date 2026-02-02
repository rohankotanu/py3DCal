import os
import torch
import pandas as pd
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
from ..lib.validate_parameters import validate_root

class ReSkinDataset(Dataset):
    """
    ReSkin Sensor Dataset.

    Args:
        root (str or pathlib.Path): The root directory that contains the dataset folder.
    """
    def __init__(self, root: Union[str, Path], subtract_no_contact=True):
        validate_dataset(root)

        self.root = root
        self.data_path = os.path.join(root, "probe_data.csv")
        self.no_contact_path = os.path.join(root, "no_contact_data.csv")

        # Load the CSV data
        self.data = pd.read_csv(self.data_path)
        self.no_contact_data = pd.read_csv(self.no_contact_path)

        # Remove temperature columns
        self.data = self.data.loc[:, ~self.data.columns.str.contains('T')]
        self.no_contact_data = self.no_contact_data.loc[:, ~self.no_contact_data.columns.str.contains('T')]

        # Load and transform blank image
        if subtract_no_contact:
            no_contact_avg = self.no_contact_data.iloc[:, 3:].mean(axis=0)
            self.data.iloc[:, 3:] = self.data.iloc[:, 3:] - no_contact_avg.values

    def __len__(self):
        return len(self.data)  # Use the DataFrame length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

            for i in idx:
                if i < 0 or i >= len(self.data):
                    raise IndexError(f"Index {i} out of range")
        else:
            # Check if index is valid
            if idx < 0 or idx >= len(self.data):
                raise IndexError("Index out of range")

        reading = torch.tensor(self.data.iloc[idx, 3:].values, dtype=torch.float32)
        target = torch.tensor(self.data.iloc[idx, 0:3].values, dtype=torch.float32)

        sample = (reading, target)

        return sample


def validate_dataset(root):
    """
    Validates the ReSkin dataset path specified by the user.

    Args:
        root: Dataset path specified by the user.
    Returns:
        None.
    Raises:
        FileNotFoundError: If necessary files do not exist.
    """
    validate_root(root)

    probe_data = os.path.join(root, "probe_data.csv")
    no_contact_data = os.path.join(root, "no_contact_data.csv")

    # Check if root directory exists
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset root directory '{root}' does not exist.")

    # Check if all the necessary files exist
    if not os.path.exists(probe_data):
        raise FileNotFoundError(f"probe_data.csv file not found in dataset root.")

    if not os.path.exists(no_contact_data):
        raise FileNotFoundError(f"no_contact_data.csv file not found in dataset root.")