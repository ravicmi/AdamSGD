"""
Data loaders for MNIST and CIFAR-10 datasets with automatic download.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os


def get_mnist_loaders(config, batch_size=128, num_workers=2):
    """
    Get MNIST data loaders with automatic download.
    
    Args:
        config (dict): Configuration dictionary with dataset settings
        batch_size (int): Batch size for training. Default: 128
        num_workers (int): Number of workers for data loading. Default: 2
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get config settings
    data_dir = config['datasets']['mnist']['data_dir']
    normalize_mean = config['datasets']['mnist']['normalize_mean']
    normalize_std = config['datasets']['mnist']['normalize_std']
    train_val_split = config['datasets']['mnist']['train_val_split']
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Split training data into train and validation
    train_size = int(train_val_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['training']['seed'])
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(config, batch_size=128, num_workers=2):
    """
    Get CIFAR-10 data loaders with automatic download and optional data augmentation.
    
    Args:
        config (dict): Configuration dictionary with dataset settings
        batch_size (int): Batch size for training. Default: 128
        num_workers (int): Number of workers for data loading. Default: 2
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get config settings
    data_dir = config['datasets']['cifar10']['data_dir']
    normalize_mean = config['datasets']['cifar10']['normalize_mean']
    normalize_std = config['datasets']['cifar10']['normalize_std']
    train_val_split = config['datasets']['cifar10']['train_val_split']
    data_augmentation = config['datasets']['cifar10']['data_augmentation']
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transforms for training (with augmentation)
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    
    # Define transforms for validation/test (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    # Download and load training data
    train_dataset_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split training data into train and validation
    train_size = int(train_val_split * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset_temp = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['training']['seed'])
    )
    
    # Create validation dataset with test transform (no augmentation)
    val_dataset_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=test_transform
    )
    
    # Get the same validation indices
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_dataset_temp.indices)
    
    # Download and load test data
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

