import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from ..datasets.tactile_sensor_dataset import TactileSensorDataset
from ..datasets.split_dataset import split_dataset
from .validate_parameters import validate_device


def train_model(model: nn.Module, dataset: TactileSensorDataset, num_epochs: int = 60, batch_size: int = 64, learning_rate: float = 1e-4, train_ratio: float = 0.8, loss_fn: nn.Module = nn.MSELoss(), device='cpu'):
    """
    Train TouchNet model on a dataset for 60 epochs with a
    64 batch size, and AdamW optimizer with learning rate 1e-4.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        dataset (py3DCal.datasets.TactileSensorDataset): The dataset to train the model on.
        num_epochs (int): Number of epochs to train for. Defaults to 60.
        batch_size (int): Batch size. Defaults to 64.
        learning_rate (float): Learning rate. Defaults to 1e-4.
        train_ratio (float): Proportion of data to use for training. Defaults to 0.8.
        loss_fn (nn.Module): Loss function. Defaults to nn.MSELoss().
        device (str): Device to run the training on. Defaults to 'cpu'.

    Outputs:
        weights.pth: Trained model weights.
        loss.csv: Training and testing losses.
    """
    validate_device(device)
    _validate_model_and_dataset(model, dataset)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=train_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True if device == "cuda" else False, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True if device == "cuda" else False, persistent_workers=True)

    model.to(device)

    epoch_train_losses = []
    epoch_val_losses = []
    
    print("Starting training...\n")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            print(f"  [Batch {batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(torch.float32).to(device)
                targets = targets.to(torch.float32).to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_val_losses.append(avg_val_loss)
        print(f"VAL LOSS: {avg_val_loss:.4f}")

    with open("losses.csv", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i in range(len(epoch_train_losses)):
            f.write(f"{i+1},{epoch_train_losses[i]},{epoch_val_losses[i]}\n")
    
    torch.save(model.state_dict(), "weights.pth")

def _validate_model_and_dataset(model: nn.Module, dataset: TactileSensorDataset):
    if not isinstance(model, nn.Module):
        raise ValueError("Model must be an instance of torch.nn.Module.")
    
    if not isinstance(dataset, TactileSensorDataset):
        raise ValueError("Dataset must be an instance of py3DCal.datasets.TactileSensorDataset.")