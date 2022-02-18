import os
from typing import Tuple

import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from eml.config import Config


def load_data(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset_full = torchvision.datasets.FashionMNIST(
        os.path.expanduser("~/dataset"),
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
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
