from pyexpat import model
import numpy as np
import torch
from pathlib import Path
from typing import Union
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from .validate_parameters import validate_device
from .add_coordinate_embeddings import add_coordinate_embeddings
from .fast_poisson import fast_poisson

def get_depthmap(model, image_path: Union[str, Path], blank_image_path: Union[str, Path], device='cpu') -> np.ndarray:
        """
        Returns the depthmap for a given input image.
        Args:
            model: A model which takes in an image and outputs gradient maps.
            image_path (str or pathlib.Path): Path to the input image.
            blank_image_path (str or pathlib.Path): Path to the blank image.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.
        Returns:
            depthmap (numpy.ndarray): The computed depthmap.
        """
        validate_device(device)

        transform = transforms.ToTensor()

        model.to(device)
        model.eval()
        image = transform(Image.open(image_path).convert("RGB"))
        blank_image = transform(Image.open(blank_image_path).convert("RGB"))
        augmented_image = image - blank_image
        augmented_image = add_coordinate_embeddings(augmented_image)
        augmented_image = augmented_image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(augmented_image)
        
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        depthmap = fast_poisson(output[:,:,0], output[:,:,1])

        depthmap = np.clip(-depthmap, a_min=0, a_max=None)

        return depthmap

def save_2d_depthmap(model, image_path: Union[str, Path], blank_image_path: Union[str, Path], device='cpu', save_path: Union[str, Path] = Path("depthmap.png")):
    """
    Save an image of the depthmap for a given input image.
    Args:
        image_path (str): Path to the input image.
        save_path (str or pathlib.Path): Path to save the depthmap image.

    Returns:
        None.
    """
    depthmap = get_depthmap(model=model, image_path=image_path, blank_image_path=blank_image_path, device=device)

    plt.imsave(save_path, depthmap, cmap='viridis')

def show_2d_depthmap(model, image_path: Union[str, Path], blank_image_path: Union[str, Path], device='cpu'):
    """
    Show the depthmap for a given input image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None.
    """
    depthmap = get_depthmap(model=model, image_path=image_path, blank_image_path=blank_image_path, device=device)

    plt.imshow(depthmap)
    plt.show()