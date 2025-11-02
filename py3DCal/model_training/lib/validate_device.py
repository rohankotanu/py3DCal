import torch

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