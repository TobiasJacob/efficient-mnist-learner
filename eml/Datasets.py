import os
from typing import Tuple

import torch
import torchvision
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from eml.Config import Config


def load_data(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Helper method to create all the dataloaders

    Args:
        cfg (Config): The global config object

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Returns three dataloaders:
                - The train data loader with all images(+labels) for the autoencoder.
                    It does contain the images as well, but they are not used by the
                    autoencoder.
                - The smaller train data loader with images+labels for the classifier.
                - The evaluation data loader with images+labels.
    """
    tra = [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10, expand=True),
        transforms.RandomResizedCrop((28, 28), scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomErasing(),
    ]
    # tra = [transforms.ToTensor(), transforms.AutoAugment()]
    train_dataset_full = torchvision.datasets.FashionMNIST(
        os.path.expanduser("~/dataset"),
        train=True,
        download=True,
        transform=transforms.Compose(tra),
    )
    train_loader_full = torch.utils.data.DataLoader(
        train_dataset_full,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
    )
    eval_dataset = torchvision.datasets.FashionMNIST(
        os.path.expanduser("~/dataset"),
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.workers
    )
    train_dataset_reduced = torch.utils.data.Subset(
        train_dataset_full, torch.arange(0, cfg.num_train_labels)
    )
    train_loader_reduced = torch.utils.data.DataLoader(
        train_dataset_reduced,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
    )
    return (train_loader_full, train_loader_reduced, eval_loader)
