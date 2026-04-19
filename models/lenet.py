"""
LeNet-5 — exact LeCun 1998 architecture for MNIST (28×28, 1-channel).

Original paper: "Gradient-Based Learning Applied to Document Recognition"
                LeCun et al., 1998.

Architecture (faithful to original):
    C1  : Conv2d(1,  6,  5×5, pad=2) → Tanh   [pad=2 keeps 28→28]
    S2  : AvgPool2d(2×2, stride=2)             [28→14]
    C3  : Conv2d(6,  16, 5×5)        → Tanh   [14→10]
    S4  : AvgPool2d(2×2, stride=2)             [10→5]
    C5  : Linear(400, 120)           → Tanh   [5×5×16=400]
    F6  : Linear(120, 84)            → Tanh
    Out : Linear(84,  num_classes)

Key authenticity notes:
  - Tanh activations (NOT ReLU) — this is what makes it LeNet-5
  - AvgPool (NOT MaxPool) — faithful to the 1998 paper
  - padding=2 on first conv so 28×28 input works (original was 32×32)

Paper target accuracy: ~99% on MNIST (non-IID FL setting ~96-98%)
"""
import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            # C1: 1 → 6 feature maps, 5×5 kernel, pad=2 to preserve 28×28
            nn.Conv2d(1, 6, kernel_size=5, padding=2),   # 28×28 → 28×28
            nn.Tanh(),
            # S2: 2×2 average pooling
            nn.AvgPool2d(kernel_size=2, stride=2),        # 28×28 → 14×14

            # C3: 6 → 16 feature maps, 5×5 kernel, no padding
            nn.Conv2d(6, 16, kernel_size=5),              # 14×14 → 10×10
            nn.Tanh(),
            # S4: 2×2 average pooling
            nn.AvgPool2d(kernel_size=2, stride=2),        # 10×10 → 5×5
        )

        self.classifier = nn.Sequential(
            # C5 (treated as FC in modern impl): 16×5×5=400 → 120
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            # F6
            nn.Linear(120, 84),
            nn.Tanh(),
            # Output
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten: 16×5×5 = 400
        x = self.classifier(x)
        return x