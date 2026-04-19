"""
ResNet-18 — paper-exact implementation for CIFAR-10 (32×32, 3-channel).

Key differences from the ImageNet ResNet-18:
  - conv1: 3×3 kernel, stride=1, padding=1  (ImageNet uses 7×7 stride-2)
  - NO MaxPool after conv1                   (would destroy 32×32 spatial dims)
  - AdaptiveAvgPool2d(1,1) before FC         (replaces hardcoded avg_pool2d(out, 4))

This is the standard CIFAR-10 ResNet used in virtually every FL paper
including the AAAI-22 multi-job FL paper (Zhou et al. 2022).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for small 32×32 inputs (CIFAR-10).

    Architecture:
        conv1  : Conv2d(3, 64, 3×3, stride=1, pad=1) → BN → ReLU   [NO maxpool]
        layer1 : 2× BasicBlock(64,  64,  stride=1)
        layer2 : 2× BasicBlock(64,  128, stride=2)
        layer3 : 2× BasicBlock(128, 256, stride=2)
        layer4 : 2× BasicBlock(256, 512, stride=2)
        pool   : AdaptiveAvgPool2d(1, 1)
        fc     : Linear(512, num_classes)

    Paper target accuracy: ~80% on CIFAR-10 (non-IID FL setting ~48-52%)
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # CIFAR-10 head: 3×3 conv, stride 1, NO maxpool
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear  = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, stride):
        layer = nn.Sequential(
            BasicBlock(self.in_planes, planes, stride),
            BasicBlock(planes, planes, 1)
        )
        self.in_planes = planes * BasicBlock.expansion
        return layer

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # 32×32 → 32×32
        out = self.layer1(out)                   # 32×32 → 32×32
        out = self.layer2(out)                   # 32×32 → 16×16
        out = self.layer3(out)                   # 16×16 →  8×8
        out = self.layer4(out)                   #  8×8  →  4×4
        out = self.avgpool(out)                  #  4×4  →  1×1
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out