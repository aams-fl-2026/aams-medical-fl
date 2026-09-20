"""
VGG-11 adapted for small inputs (28x28 grayscale EMNIST-Letters).
Removes last MaxPool to prevent spatial collapse on 28x28 input.
"""
import torch
import torch.nn as nn


class VGG11(nn.Module):
    def __init__(self, num_classes=26, input_channels=1):
        super(VGG11, self).__init__()

        def conv_bn_relu(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn_relu(input_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 28→14
            conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 14→7
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 7→3
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 3→1
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            # NO final MaxPool — would collapse 1x1 to 0x0
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = VGG11(num_classes=26, input_channels=1)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("VGG11 working correctly!")