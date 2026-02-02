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
            model: A model which takes in an image and outputs gradient maps.
            sensor_reading (torch.tensor, numpy.ndarray, list): The sensor reading input.
            no_contact_reading (torch.tensor, numpy.ndarray, list, optional): The no contact reading to subtract from the sensor reading. Defaults to None.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.

        Returns:
            depthmap (numpy.ndarray): The computed depthmap.
        """
        validate_device(device)

        model.to(device)
        model.eval()

        if isinstance(sensor_reading, (np.ndarray, list)):
            sensor_reading = torch.tensor(sensor_reading, dtype=torch.float32)

        if no_contact_reading is not None:
            if isinstance(no_contact_reading, (np.ndarray, list)):
                no_contact_reading = torch.tensor(no_contact_reading, dtype=torch.float32)

            sensor_reading = sensor_reading - no_contact_reading
            
        sensor_reading = sensor_reading.to(device)

        with torch.no_grad():
            output = model(sensor_reading)

        return output.cpu().numpy()