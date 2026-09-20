"""
Federated Learning Client
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy


class FLClient:
    def __init__(self, device_id, local_dataset, batch_size=32,
                 learning_rate=0.01, device='cpu'):
        self.device_id    = device_id
        self.local_dataset = local_dataset
        self.batch_size   = batch_size
        self.learning_rate = learning_rate
        self.device       = torch.device(device) if isinstance(device, str) else device

        self.train_loader = DataLoader(
            local_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        print(f"Client {device_id}: {len(local_dataset)} samples")

    def train(self, global_model, epochs=5):
        """
        Train model on local data.
        Improvements over baseline:
          - Momentum=0.9 + weight_decay=1e-4 (standard for ResNet/CIFAR-10)
          - Cosine annealing LR schedule within local epochs
            (reduces oscillation, improves convergence)
        """
        model = deepcopy(global_model).to(self.device)
        model.train()

        # SGD with momentum and weight decay — standard for FL
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Cosine annealing within local epochs — smoother LR decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        criterion = nn.CrossEntropyLoss()

        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss    = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss   = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss    += loss.item() * data.size(0)
                pred           = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
                epoch_samples += data.size(0)

            scheduler.step()

            total_loss    += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

        avg_loss     = total_loss    / total_samples if total_samples > 0 else 0.0
        avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

        return {
            'model_state': model.state_dict(),
            'loss':        avg_loss,
            'accuracy':    avg_accuracy,
            'num_samples': len(self.local_dataset)
        }