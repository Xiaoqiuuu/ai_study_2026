"""Dataset and DataLoader builders for MNIST."""
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create MNIST train/val/test DataLoaders with standard normalization.

    Args:
        data_dir: Root directory to store MNIST data.
        batch_size: Batch size for all loaders.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, copy tensors into CUDA pinned memory.
        val_ratio: Fraction of training data to use for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Standard MNIST preprocessing: flatten to 784-dim vector, normalize to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image [0,255] -> Float [0,1]
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten 28x28 -> 784
    ])

    # Download full training set
    full_train = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Split train / val
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_set = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # DataLoaders
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_set, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader
