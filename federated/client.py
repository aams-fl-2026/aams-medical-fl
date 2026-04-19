"""
Federated Learning Client

Handles local training on device
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy

class FLClient:
    """
    Federated Learning Client (Device)
    
    Responsibilities:
    - Receive global model from server
    - Train on local data
    - Send updated model back to server
    """
    
    def __init__(self, device_id, local_dataset, batch_size=32, 
                 learning_rate=0.01, device='cpu'):
        """
        Args:
            device_id: Unique device identifier
            local_dataset: PyTorch Dataset for this device
            batch_size: Batch size for training
            learning_rate: Learning rate
            device: 'cuda' or 'cpu'
        """
        self.device_id = device_id
        self.local_dataset = local_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        # Create data loader
        self.train_loader = DataLoader(
            local_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        print(f"Client {device_id}: {len(local_dataset)} samples")
    
    def train(self, global_model, epochs=5):
        """
        Train model on local data
        
        Args:
            global_model: Model received from server
            epochs: Number of local epochs
            
        Returns:
            dict: Updated model state_dict and training stats
        """
        # Copy global model
        model = deepcopy(global_model).to(self.device)
        model.train()
        
        # Optimizer and loss
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Local training
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
                epoch_samples += data.size(0)
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        # Final stats
        avg_loss = total_loss / total_samples
        avg_accuracy = 100.0 * total_correct / total_samples
        
        return {
            'model_state': model.state_dict(),
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'num_samples': len(self.local_dataset)
        }


# Test
if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath('..'))
    sys.path.insert(0, os.path.abspath('.'))
    
    from data.non_iid_partition import create_non_iid_datasets
    from models.cnn import SimpleCNN    
    print("Testing FL Client\n")
    print("=" * 60)
    
    # Get data for one device
    print("Creating non-IID datasets...")
    device_datasets, _ = create_non_iid_datasets('cifar10', num_devices=5)
    
    # Create client
    print("\nCreating FL Client...")
    client = FLClient(
        device_id=0,
        local_dataset=device_datasets[0],
        batch_size=32,
        learning_rate=0.01
    )
    
    # Create model
    model = SimpleCNN(num_classes=10)
    
    # Train
    print("\nTraining for 2 epochs...")
    print("(This will take ~30 seconds)")
    result = client.train(model, epochs=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Loss: {result['loss']:.4f}")
    print(f"Accuracy: {result['accuracy']:.2f}%")
    print(f"Samples: {result['num_samples']}")
    print("\nFL Client working correctly!")