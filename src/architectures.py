import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size=256, kernel_sz=3):
        super(SimpleCNN, self).__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=kernel_sz, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_sz, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (patch_size // 2) * (patch_size // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x