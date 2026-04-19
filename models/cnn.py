"""
Simple CNN for CIFAR-10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10
    
    Architecture:
    - Conv1: 3->32 channels, 3x3 kernel
    - Conv2: 32->64 channels, 3x3 kernel  
    - Conv3: 64->64 channels, 3x3 kernel
    - FC1: 512 units
    - FC2: 10 classes (output)
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1: 32x32 -> 16x16
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2: 16x16 -> 8x8
        x = self.pool(F.relu(self.conv2(x)))
        
        # Conv block 3: 8x8 -> 4x4
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 64 * 4 * 4)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Test
if __name__ == "__main__":
    print("Testing SimpleCNN\n")
    
    model = SimpleCNN(num_classes=10)
    
    # Test with random input (batch of 4 CIFAR-10 images)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nCNN model working correctly!")