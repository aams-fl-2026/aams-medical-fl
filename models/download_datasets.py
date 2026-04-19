"""
Download all datasets used in the paper
Group A: CIFAR-10, EMNIST-Letters, EMNIST-Digits
Group B: CIFAR-10, Fashion-MNIST, MNIST
"""
import torch
from torchvision import datasets, transforms
import os

def download_datasets(data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)

    # CIFAR-10
    print("Downloading CIFAR-10...")
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=transform_cifar)
    datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_cifar)

    # MNIST
    print("Downloading MNIST...")
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    datasets.MNIST(root=data_dir, train=True,  download=True, transform=transform_mnist)
    datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_mnist)

    # Fashion-MNIST
    print("Downloading Fashion-MNIST...")
    datasets.FashionMNIST(root=data_dir, train=True,  download=True, transform=transform_mnist)
    datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_mnist)


    print("\nAll datasets downloaded successfully!")

if __name__ == "__main__":
    download_datasets()
