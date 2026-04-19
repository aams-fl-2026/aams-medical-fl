"""
Federated Learning Server
Coordinates the training process:
- Manages global model
- Selects devices for each round
- Aggregates local updates
- Evaluates global model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from collections import OrderedDict


class FLServer:
    """
    Federated Learning Server
    Responsibilities:
    - Initialize and maintain global model
    - Coordinate training rounds
    - Aggregate updates from clients (FedAvg)
    - Evaluate on test set
    """

    def __init__(self, model, test_dataset, device='cpu'):
        """
        Args:
            model: Global model to train
            test_dataset: Test dataset for evaluation
            device: 'cuda' or 'cpu'
        """
        self.global_model = model
        self.test_dataset = test_dataset
        self.device = torch.device(device)

        # Move model to device
        self.global_model.to(self.device)

        # Test data loader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'selected_devices': []
        }

        print(f"FL Server initialized")
        print(f"  Test dataset: {len(test_dataset)} samples")
        print(f"  Device: {self.device}")

    def aggregate(self, client_updates):
        """
        FedAvg aggregation: Weighted average of client models
        Args:
            client_updates: List of dicts from clients
                Each dict has: {'model_state', 'num_samples', 'loss', 'accuracy'}
        Returns:
            dict: Aggregated training statistics
        """
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in client_updates)

        # Initialize aggregated state dict
        aggregated_state = OrderedDict()

        # Get first client's state dict structure
        first_state = client_updates[0]['model_state']

        # For each parameter in the model
        for key in first_state.keys():
            if first_state[key].dtype.is_floating_point:
                # Weighted average for float parameters (weights, biases, BN running stats)
                aggregated_state[key] = torch.zeros_like(first_state[key])
                for update in client_updates:
                    weight = update['num_samples'] / total_samples
                    aggregated_state[key] += update['model_state'][key] * weight
            else:
                # Integer tensors like num_batches_tracked — copy from first client
                aggregated_state[key] = first_state[key].clone()

        # Update global model
        self.global_model.load_state_dict(aggregated_state)

        # Calculate weighted average of metrics
        avg_loss = sum(
            update['loss'] * update['num_samples']
            for update in client_updates
        ) / total_samples

        avg_accuracy = sum(
            update['accuracy'] * update['num_samples']
            for update in client_updates
        ) / total_samples

        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'num_clients': len(client_updates),
            'total_samples': total_samples
        }

    def evaluate(self):
        """
        Evaluate global model on test set
        Returns:
            dict: Test accuracy and loss
        """
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

        avg_loss = test_loss / total
        accuracy = 100.0 * correct / total

        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy
        }

    def get_global_model(self):
        """Return copy of current global model"""
        return copy.deepcopy(self.global_model)

    def save_checkpoint(self, path, round_num):
        """Save model checkpoint"""
        torch.save({
            'round': round_num,
            'model_state_dict': self.global_model.state_dict(),
            'history': self.history
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from round {checkpoint['round']}")


# Test
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data.non_iid_partition import create_non_iid_datasets
    from models.cnn import SimpleCNN
    from federated.client import FLClient

    print("Testing FL Server\n")
    print("=" * 60)

    # Create data
    print("Creating datasets...")
    device_datasets, test_dataset = create_non_iid_datasets('cifar10', num_devices=3)

    # Create model
    model = SimpleCNN(num_classes=10)

    # Create server
    print("\nCreating FL Server...")
    server = FLServer(model, test_dataset, device='cpu')

    # Create clients
    print("\nCreating clients...")
    clients = []
    for i in range(3):
        client = FLClient(
            device_id=i,
            local_dataset=device_datasets[i],
            batch_size=32
        )
        clients.append(client)

    # Simulate one round
    print("\n" + "=" * 60)
    print("SIMULATING ONE TRAINING ROUND")
    print("=" * 60)

    # Get global model
    global_model = server.get_global_model()

    # Each client trains
    print("\nClients training locally...")
    client_updates = []
    for client in clients:
        update = client.train(global_model, epochs=1)
        client_updates.append(update)
        print(f"  Client {client.device_id}: Loss={update['loss']:.4f}, "
              f"Acc={update['accuracy']:.2f}%")

    # Server aggregates
    print("\nServer aggregating updates...")
    agg_stats = server.aggregate(client_updates)
    print(f"  Aggregated Loss: {agg_stats['loss']:.4f}")
    print(f"  Aggregated Accuracy: {agg_stats['accuracy']:.2f}%")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_stats = server.evaluate()
    print(f"  Test Accuracy: {test_stats['test_accuracy']:.2f}%")

    print("\n" + "=" * 60)
    print("FL Server working correctly!")
    print("=" * 60)