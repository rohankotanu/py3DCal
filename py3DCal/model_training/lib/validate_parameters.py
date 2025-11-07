import os
import torch
from typing import Union
from pathlib import Path

def validate_device(device: str):
        """
        Validates the device by converting it to a torch.device object.
        Args:
            device (str): Device to run the model on.
        Returns:
            None.
        Raises:
            ValueError: If the device is not specified or invalid.
        """
        try:
            device = torch.device(device)
        except Exception as e:
            raise ValueError(
                f"Invalid device '{device}'. Valid options include:\n"
                "  - 'cpu': CPU processing\n"
                "  - 'cuda' or 'cuda:0': NVIDIA GPU\n"
                "  - 'mps': Apple Silicon GPU\n"
                "See: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device"
            ) from e
        
def validate_root(root, must_exist=False):
    """
    Validates the root path specified by the user.

    Args:
        root: root path specified by the user.
    Returns:
        None.
    Raises:
        ValueError: If the root is not specified or invalid.
    """
    if root is None :
       raise ValueError(
           "root directory cannot be None.\n"
       )
    
    if not isinstance(root, (str, Path)):
       raise ValueError(
           "root directory must be a valid file system path as a string or pathlib.Path object\n"
       )
    
    if must_exist and not os.path.exists(root):
         raise ValueError(
              f"root directory '{root}' does not exist.\n"
         )

def validate_dataset(root, subtract_blank: bool):
    """
    Validates the dataset path specified by the user.

    Args:
        root: Dataset path specified by the user.
        subtract_blank (bool): Whether to subtract blank image.
    Returns:
        None.
    Raises:
        FileNotFoundError: If necessary files do not exist.
    """
    validate_root(root)

    annotation_path = os.path.join(root, "annotations", "annotations.csv")
    metadata_path = os.path.join(root, "annotations", "metadata.json")
    probe_images_path = os.path.join(root, "probe_images")
    blank_image_path = os.path.join(root, "blank_images", "blank.png")

    # Check if root directory exists
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset root directory '{root}' does not exist.")

    # Check if all the necessary files exist
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"annotations.csv file not found in annotations/ directory. Use py3DCal.annotations() function to create it.")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json file not found in annotations/ directory. Use py3DCal.annotations() function to create it.")

    if not os.path.exists(probe_images_path):
        raise FileNotFoundError(f"probe_images/ directory not found in dataset root.")

    if subtract_blank and not os.path.exists(blank_image_path):
        raise FileNotFoundError(f"blank.png file not found in blank_images/ directory.")