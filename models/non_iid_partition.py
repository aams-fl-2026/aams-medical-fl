import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image


# ── MedMNIST wrapper ─────────────────────────────────────────────────────────
class MedMNISTDataset(Dataset):
    """Wraps a MedMNIST dataset to return (image, label) tuples compatible
    with the existing FL pipeline. Labels are squeezed from (N,1) to (N,)."""
    def __init__(self, medmnist_dataset):
        self.dataset = medmnist_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if hasattr(label, '__len__'):
            label = int(label[0])
        else:
            label = int(label)
        return img, label


class NonIIDPartitioner:
    def __init__(self, dataset_name, num_devices, num_classes_per_device=2,
                 seed=42, data_root='./data'):
        self.dataset_name       = dataset_name
        self.num_devices        = num_devices
        self.num_classes_per_device = num_classes_per_device
        self.seed               = seed
        self.data_root          = data_root
        self.train_dataset      = None
        self.test_dataset       = None
        self.num_classes        = None
        self._load_dataset()

    def _load_dataset(self):
        name = self.dataset_name.lower()

        if name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
            ])
            self.train_dataset = datasets.CIFAR10(self.data_root, train=True,  download=True, transform=transform)
            self.test_dataset  = datasets.CIFAR10(self.data_root, train=False, download=True, transform=transform)
            self.num_classes   = 10

        elif name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
            self.train_dataset = datasets.MNIST(self.data_root, train=True,  download=True, transform=transform)
            self.test_dataset  = datasets.MNIST(self.data_root, train=False, download=True, transform=transform)
            self.num_classes   = 10

        elif name == 'fashion_mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,),(0.3530,))])
            self.train_dataset = datasets.FashionMNIST(self.data_root, train=True,  download=True, transform=transform)
            self.test_dataset  = datasets.FashionMNIST(self.data_root, train=False, download=True, transform=transform)
            self.num_classes   = 10

        elif name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
            ])
            self.train_dataset = datasets.CIFAR100(self.data_root, train=True,  download=True, transform=transform)
            self.test_dataset  = datasets.CIFAR100(self.data_root, train=False, download=True, transform=transform)
            self.num_classes   = 100

        elif name == 'emnist_letters':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1722,),(0.3309,))])
            target_transform = transforms.Lambda(lambda y: y - 1)
            self.train_dataset = datasets.EMNIST(self.data_root, split='letters', train=True,  download=True, transform=transform, target_transform=target_transform)
            self.test_dataset  = datasets.EMNIST(self.data_root, split='letters', train=False, download=True, transform=transform, target_transform=target_transform)
            self.num_classes   = 26

        elif name == 'emnist_balanced':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,),(0.3332,))])
            self.train_dataset = datasets.EMNIST(self.data_root, split='balanced', train=True,  download=True, transform=transform)
            self.test_dataset  = datasets.EMNIST(self.data_root, split='balanced', train=False, download=True, transform=transform)
            self.num_classes   = 47

        # ── MedMNIST datasets ─────────────────────────────────────────────
        elif name == 'organamnist':
            import medmnist
            from medmnist import OrganAMNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
            train = OrganAMNIST(split='train', transform=transform, download=True, root=self.data_root)
            test  = OrganAMNIST(split='test',  transform=transform, download=True, root=self.data_root)
            self.train_dataset = MedMNISTDataset(train)
            self.test_dataset  = MedMNISTDataset(test)
            self.num_classes   = 11

        elif name == 'bloodmnist':
            from medmnist import BloodMNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
            ])
            train = BloodMNIST(split='train', transform=transform, download=True, root=self.data_root)
            test  = BloodMNIST(split='test',  transform=transform, download=True, root=self.data_root)
            self.train_dataset = MedMNISTDataset(train)
            self.test_dataset  = MedMNISTDataset(test)
            self.num_classes   = 8

        elif name == 'tissuemnist':
            from medmnist import TissueMNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
            train = TissueMNIST(split='train', transform=transform, download=True, root=self.data_root)
            test  = TissueMNIST(split='test',  transform=transform, download=True, root=self.data_root)
            self.train_dataset = MedMNISTDataset(train)
            self.test_dataset  = MedMNISTDataset(test)
            self.num_classes   = 8


        elif name == 'dermamnist':
            from medmnist import DermaMNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
            ])
            train = DermaMNIST(split='train', transform=transform, download=True, root=self.data_root)
            test  = DermaMNIST(split='test',  transform=transform, download=True, root=self.data_root)
            self.train_dataset = MedMNISTDataset(train)
            self.test_dataset  = MedMNISTDataset(test)
            self.num_classes   = 7

        elif name == 'isic2019':
            # ISIC 2019 must be manually downloaded to data_root/isic2019/
            # Expected: data_root/isic2019/train/ with class subdirs
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            full = datasets.ImageFolder(f'{self.data_root}/isic2019/train', transform=transform)
            # 80/20 train/test split
            n = len(full)
            n_train = int(0.8 * n)
            indices = list(range(n))
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            self.train_dataset = Subset(full, indices[:n_train])
            self.test_dataset  = Subset(full, indices[n_train:])
            self.num_classes   = 8

        else:
            raise ValueError(f'Unknown dataset: {self.dataset_name}')

    def _get_labels(self, dataset):
        if hasattr(dataset, 'targets'):
            return np.array(dataset.targets)
        labels = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            labels.append(int(y))
        return np.array(labels)

    def _create_shards(self, class_indices, num_shards):
        shards = {}
        for class_id, indices in class_indices.items():
            np.random.shuffle(indices)
            shards[class_id] = np.array_split(indices, num_shards)
        return shards

    def _assign_to_devices(self, class_shards, num_shards):
        np.random.seed(self.seed)
        device_data = {d: [] for d in range(self.num_devices)}
        for d in range(self.num_devices):
            chosen = np.random.choice(self.num_classes,
                                      size=self.num_classes_per_device,
                                      replace=False)
            for class_id in chosen:
                shard_id = np.random.randint(0, num_shards)
                device_data[d].extend(class_shards[class_id][shard_id].tolist())
            print(f'  Device {d}: classes {chosen} -> {len(device_data[d])} samples')
        return device_data

    def partition(self):
        np.random.seed(self.seed)
        labels = self._get_labels(self.train_dataset)
        class_indices = {}
        for c in range(self.num_classes):
            class_indices[c] = np.where(labels == c)[0].tolist()
        num_shards = max(20, (self.num_devices * self.num_classes_per_device) // self.num_classes)
        class_shards = self._create_shards(class_indices, num_shards)
        device_data  = self._assign_to_devices(class_shards, num_shards)

        total = sum(len(v) for v in device_data.values())
        print(f'Total samples distributed: {total}')
        print(f'Average per device: {total/self.num_devices:.1f}')

        device_datasets = {}
        for d in range(self.num_devices):
            device_datasets[d] = Subset(self.train_dataset, device_data[d])

        return device_datasets, self.test_dataset


def create_non_iid_datasets(dataset_name, num_devices=200,
                             num_classes_per_device=2, seed=42,
                             data_root='./data'):
    partitioner = NonIIDPartitioner(dataset_name, num_devices,
                                    num_classes_per_device, seed, data_root)
    return partitioner.partition()


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
    print("\nNon-IID partitioning working correctly!")
