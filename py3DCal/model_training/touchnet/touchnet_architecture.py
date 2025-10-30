import torch.nn as nn
import torch.nn.functional as F

class TouchNetArchitecture(nn.Module):
    """
    TouchNetArchitecture: A PyTorch neural network architecture for TouchNet.
    """
    def __init__(self):
        super().__init__()

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