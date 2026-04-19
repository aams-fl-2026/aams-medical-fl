"""
CNN-B — exact architecture from Zhou et al. AAAI-22 appendix.
Two 2x2 conv layers (64, 32 channels) with dropout 0.05.
Used in Group B with Fashion-MNIST (28x28, 1-channel).
"""
import torch.nn as nn
import torch.nn.functional as F

class CNNB(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNB, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2)
        self.drop1 = nn.Dropout(0.05)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=2)
        self.drop2 = nn.Dropout(0.05)
        self.pool  = nn.AdaptiveAvgPool2d((6, 6))
        self.fc    = nn.Linear(32 * 6 * 6, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
