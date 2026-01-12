import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGateCNN(nn.Module):
    """
    Lightweight CNN with Batch Normalization.
    Structure: [Conv -> BN -> ReLU -> Pool] x 3 -> Flatten -> FC -> Dropout -> FC
    """

    def __init__(self, dropout_prob=0.5):
        super(SimpleGateCNN, self).__init__()

        # --- Block 1 ---
        # Input: (Batch, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # <--- NEW: Normalize the 16 channels

        # --- Block 2 ---
        # Input: (Batch, 16, 32, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # <--- NEW: Normalize the 32 channels

        # --- Block 3 ---
        # Input: (Batch, 32, 16, 16)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # <--- NEW: Normalize the 64 channels

        # Pooling layer (reused)
        self.pool = nn.MaxPool2d(2, 2)

        # --- Classification Head ---
        # Flattened size: 64 channels * 8 height * 8 width
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten: Turn 3D volume into 1D vector
        x = x.view(x.size(0), -1)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
