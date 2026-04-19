"""
Non-IID data partitioning for federated learning
"""
import numpy as np
import torch
from torchvision import datasets, transforms
from collections import defaultdict
from torch.utils.data import Subset

class NonIIDPartitioner:
    """Partition dataset in non-IID manner"""
    
    def __init__(self, dataset, num_devices=30, num_classes=10, 
                 num_classes_per_device=2, seed=42):
        self.dataset = dataset
        self.num_devices = num_devices
        self.num_classes = num_classes
        self.num_classes_per_device = num_classes_per_device
        self.seed = seed
        np.random.seed(seed)
        
    def partition(self):
        """Partition dataset across devices in non-IID manner"""
        print(f"Partitioning dataset for {self.num_devices} devices...")
        
        # Step 1: Organize data by class
        class_indices = self._organize_by_class()
        
        # Step 2: Divide each class into shards
        num_shards = 20
        class_shards = self._create_shards(class_indices, num_shards)
        
        # Step 3: Assign shards to devices
        device_data = self._assign_to_devices(class_shards, num_shards)
        
        # Verify partitioning
        self._verify_partition(device_data)
        
        return device_data
    
    def _organize_by_class(self):
        """Group dataset indices by class label"""
        class_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            class_indices[label].append(idx)
        
        print(f"Organized {len(class_indices)} classes")
        return class_indices
    
    def _create_shards(self, class_indices, num_shards):
        """Divide each class into shards"""
        class_shards = {}
        
        for class_id, indices in class_indices.items():
            np.random.shuffle(indices)
            shards = np.array_split(indices, num_shards)
            class_shards[class_id] = shards
            
            print(f"  Class {class_id}: {len(indices)} samples -> "
                  f"{num_shards} shards of ~{len(shards[0])} samples each")
        
        return class_shards
    
    def _assign_to_devices(self, class_shards, num_shards):
        """Assign shards to devices"""
        device_data = {}
        
        for device_id in range(self.num_devices):
            selected_classes = np.random.choice(
                self.num_classes,
                self.num_classes_per_device,
                replace=False
            )
            
            device_indices = []
            for class_id in selected_classes:
                shard_id = np.random.randint(0, num_shards)
                device_indices.extend(class_shards[class_id][shard_id])
            
            device_data[device_id] = device_indices
            
            print(f"  Device {device_id}: classes {selected_classes} -> "
                  f"{len(device_indices)} samples")
        
        return device_data
    
    def _verify_partition(self, device_data):
        """Verify partitioning statistics"""
        print("\n" + "=" * 60)
        print("PARTITION VERIFICATION")
        print("=" * 60)
        
        total_samples = sum(len(indices) for indices in device_data.values())
        avg_samples = total_samples / self.num_devices
        
        print(f"Total samples distributed: {total_samples}")
        print(f"Average per device: {avg_samples:.1f}")
        
        print("\nSample device class distributions:")
        for device_id in range(min(3, self.num_devices)):
            indices = device_data[device_id]
            labels = [self.dataset[i][1] for i in indices]
            unique_classes = np.unique(labels)
            print(f"  Device {device_id}: classes {unique_classes}, "
                  f"{len(indices)} samples")
        
        print("=" * 60)
    
    def get_device_dataset(self, device_id, device_data):
        """Get Subset dataset for a specific device"""
        indices = device_data[device_id]
        return Subset(self.dataset, indices)


def create_non_iid_datasets(dataset_name='cifar10', num_devices=30, seed=42 , num_classes_per_device=2):
    """
    Convenience function to create non-IID datasets
    
    Returns:
        tuple: (device_datasets, test_dataset)
    """
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        num_classes = 10
        
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        num_classes = 10
    elif dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
        num_classes = 10
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    partitioner = NonIIDPartitioner(
        train_dataset,
        num_devices=num_devices,
        num_classes=num_classes,
        num_classes_per_device=num_classes_per_device,
        seed=seed
   )
    device_data = partitioner.partition()
    
    device_datasets = {}
    for device_id in range(num_devices):
        device_datasets[device_id] = partitioner.get_device_dataset(
            device_id, device_data
        )
    
    return device_datasets, test_dataset


if __name__ == "__main__":
    print("Testing Non-IID Partitioning\n")
    
    print("=" * 60)
    print("CIFAR-10 Partitioning")
    print("=" * 60)
    device_datasets, test_dataset = create_non_iid_datasets(
        'cifar10', num_devices=30
    )
    
    print(f"\nCreated datasets for {len(device_datasets)} devices")
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("\nVerifying Device 0:")
    device_0_data = device_datasets[0]
    print(f"  Dataset size: {len(device_0_data)}")
    
    from collections import Counter
    labels = [device_0_data[i][1] for i in range(len(device_0_data))]
    class_dist = Counter(labels)
    print(f"  Class distribution: {dict(class_dist)}")
    print(f"  Number of unique classes: {len(class_dist)}")
    
    assert len(class_dist) == 2, "Device should have exactly 2 classes!"
    print("\nNon-IID partitioning working correctly!")
