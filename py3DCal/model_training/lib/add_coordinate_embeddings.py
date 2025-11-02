import torch

def add_coordinate_embeddings(image):
        """
        Add coordinate embeddings to the input image.
        - X channel: column indices (1s in first column, 2s in second column, etc.)
        - Y channel: row indices (1s in first row, 2s in second row, etc.)

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Image tensor with added coordinate embeddings of shape (C+2, H, W).
        """
        # Get image dimensions
        _, height, width = image.shape
        
        # Create x coordinate channel (column indices)
        x_embedding = torch.arange(1, width + 1, dtype=torch.float32).unsqueeze(0).repeat(height, 1)
        x_channel = x_embedding.unsqueeze(0)  # Add channel dimension

        # Create y coordinate channel (row indices)
        y_embedding = torch.arange(1, height + 1, dtype=torch.float32).unsqueeze(1).repeat(1, width)
        y_channel = y_embedding.unsqueeze(0)  # Add channel dimension

        # Concatenate original image with position embeddings
        image_with_embeddings = torch.cat([image, x_channel, y_channel], dim=0)

        return image_with_embeddings