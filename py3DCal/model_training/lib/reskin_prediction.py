from pyexpat import model
import numpy as np
import torch
from torch import nn
from pathlib import Path
from typing import Union
from .validate_parameters import validate_device

def get_reskin_contact(model: nn.Module, sensor_reading: Union[torch.tensor, np.ndarray, list], no_contact_reading: Union[torch.tensor, np.ndarray, list]=None, device='cpu') -> np.ndarray:
        """
        Returns the depthmap for a given input image.
        Args:
            model (torch.nn.Module): A model which takes in an image and outputs gradient maps.
            sensor_reading (torch.tensor, numpy.ndarray, or list): The sensor reading input.
            no_contact_reading (torch.tensor, numpy.ndarray, or list; optional): The no contact reading to subtract from the sensor reading. Defaults to None.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.

        Returns:
            depthmap (numpy.ndarray): The computed depthmap.
        """
        validate_inputs(model, sensor_reading, no_contact_reading, device)
        
        model.to(device)
        model.eval()

        # Convert to tensor if necessary
        if isinstance(sensor_reading, (np.ndarray, list)):
            sensor_reading = torch.tensor(sensor_reading, dtype=torch.float32)

        if no_contact_reading is not None:
            # Convert to tensor if necessary
            if isinstance(no_contact_reading, (np.ndarray, list)):
                no_contact_reading = torch.tensor(no_contact_reading, dtype=torch.float32)

            # Get difference from no contact reading
            sensor_reading = sensor_reading - no_contact_reading
        
        # Get output of model
        sensor_reading = sensor_reading.to(device)

        with torch.no_grad():
            output = model(sensor_reading)

        return output.cpu().numpy()

def validate_inputs(model: nn.Module, sensor_reading: Union[torch.tensor, np.ndarray, list], no_contact_reading: Union[torch.tensor, np.ndarray, list]=None, device='cpu'):
    """
    Validates the inputs for the reskin contact prediction function.
    Args:
        model (torch.nn.Module): A model which takes in an image and outputs gradient maps.
        sensor_reading (torch.tensor, numpy.ndarray, or list): The sensor reading input.
        no_contact_reading (torch.tensor, numpy.ndarray, or list; optional): The no contact reading to subtract from the sensor reading. Defaults to None.
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """
    validate_device(device)
    
    # Validate model
    if not isinstance(model, nn.Module):
        raise TypeError("Model must be a PyTorch nn.Module.")

    # Validate sensor reading
    if not isinstance(sensor_reading, (torch.Tensor, np.ndarray, list)):
        raise TypeError("Sensor reading must be a torch.Tensor, numpy.ndarray, or list.")

    if no_contact_reading is not None:
        # Validate no contact reading
        if not isinstance(no_contact_reading, (torch.Tensor, np.ndarray, list)):
            raise TypeError("No contact reading must be a torch.Tensor, numpy.ndarray, or list.")