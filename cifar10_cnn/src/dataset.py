"""CIFAR-10 dataset with standard augmentation pipeline."""
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# CIFAR-10 官方统计的 mean / std
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create CIFAR-10 train/val/test DataLoaders.

    Training augmentation:
        - RandomCrop(32, padding=4)
        - RandomHorizontalFlip
        - ToTensor + Normalize

    Test/Val:
        - ToTensor + Normalize only
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    # 注意：val 集应该用 test_transform，不要带 augmentation
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # 给 val_set 换上无增强的 transform（hacky but works）
    val_set.dataset.transform = test_transform  # type: ignore

    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    return train_loader, val_loader, test_loader
