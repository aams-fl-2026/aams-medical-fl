"""
AlexNet — Krizhevsky et al. 2012, adapted for small inputs (28×28, 1-channel MNIST).

Original paper: "ImageNet Classification with Deep Convolutional Neural Networks"
                Krizhevsky, Sutskever, Hinton. NeurIPS 2012.

The paper (Zhou et al. AAAI-22) uses AlexNet on MNIST. Because the original
AlexNet targets 224×224 ImageNet images, the following faithful adaptations
are standard in FL literature for 28×28 inputs:

  - Conv1: 11×11/stride-4 → 5×5/stride-1  (avoids collapsing 28px to near-zero)
  - Conv2: 5×5 kept
  - Conv3/4/5: 3×3 kept (faithful to original)
  - MaxPool after conv1 removed (spatial dims too small)
  - MaxPool after conv2 kept
  - MaxPool after conv5 kept
  - AdaptiveAvgPool2d(2,2) before FC → robust to any input size
  - FC dims scaled down proportionally: 4096→2048 (input is 4× smaller)
  - Dropout(0.5) in FC layers — faithful to original
  - input_channels=1 for MNIST, input_channels=3 for RGB

Architecture for 28×28 input:
    Conv(1, 64, 5×5, s=1, p=2) → ReLU                 [28→28]
    MaxPool(3×3, s=2)                                   [28→13]
    Conv(64, 192, 3×3, p=1)    → ReLU                  [13→13]
    MaxPool(3×3, s=2)                                   [13→6]
    Conv(192, 384, 3×3, p=1)   → ReLU                  [6→6]
    Conv(384, 256, 3×3, p=1)   → ReLU                  [6→6]
    Conv(256, 256, 3×3, p=1)   → ReLU                  [6→6]
    MaxPool(3×3, s=2)                                   [6→2]
    AdaptiveAvgPool(2,2)                                [2→2]
    Dropout(0.5) → Linear(1024, 2048) → ReLU
    Dropout(0.5) → Linear(2048, 2048) → ReLU
    Linear(2048, num_classes)

Paper target accuracy: ~99% on MNIST (non-IID FL setting ~97-99%)
"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=1, input_size=28):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # Conv1: 5×5, stride=1 (adapted from 11×11/stride-4 for 28px input)
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # 28→13

            # Conv2: 3×3 (adapted from 5×5, preserves structure)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # 13→6

            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # 6→2
        )

        # Adaptive pool collapses any remaining spatial size to 2×2
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 2 * 2, 2048),       # 1024 → 2048
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x