"""
Medical FL Models — ResNet-18 based for MedMNIST datasets (28x28)
Uses the same ResNet-18 backbone as the MedMNIST benchmark paper
(Yang et al., IEEE ISBI 2023) which is the standard for these datasets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18Medical(nn.Module):
    """
    ResNet-18 for MedMNIST 28x28 images.
    Same architecture as MedMNIST benchmark (Yang et al., IEEE ISBI 2023).
    Adapted for small inputs: no maxpool after first conv.
    """
    def __init__(self, num_classes=11, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Aliases for each job — same architecture, different num_classes/channels
def ResNet50Medical(num_classes=11, input_channels=1):
    """OrganAMNIST: 11 classes, grayscale"""
    return ResNet18Medical(num_classes=num_classes, input_channels=input_channels)

def DenseNet121Medical(num_classes=8, input_channels=3):
    """BloodMNIST: 8 classes, RGB"""
    return ResNet18Medical(num_classes=num_classes, input_channels=input_channels)

def VGG16Medical(num_classes=8, input_channels=1):
    """TissueMNIST: 8 classes, grayscale"""
    return ResNet18Medical(num_classes=num_classes, input_channels=input_channels)

def DenseNet169Medical(num_classes=7, input_channels=3):
    """DermaMNIST: 7 classes, RGB"""
    return ResNet18Medical(num_classes=num_classes, input_channels=input_channels)


if __name__ == '__main__':
    for fn, ch, nc, name in [
        (ResNet50Medical,    1, 11, 'ResNet18/OrganAMNIST'),
        (DenseNet121Medical, 3,  8, 'ResNet18/BloodMNIST'),
        (VGG16Medical,       1,  8, 'ResNet18/TissueMNIST'),
        (DenseNet169Medical, 3,  7, 'ResNet18/DermaMNIST'),
    ]:
        m = fn(num_classes=nc, input_channels=ch)
        x = torch.randn(2, ch, 28, 28)
        out = m(x)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name}: output={out.shape}  params={params:.1f}M")
