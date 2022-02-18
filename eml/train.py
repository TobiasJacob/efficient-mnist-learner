import os

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms

from eml.callbacks.VisualizeEmbeddings import VisualizeEmbeddings
from eml.config import Config
from eml.networks.AutoEncoder import AutoEncoder
from eml.networks.Classifier import Classifier


def train(cfg: Config) -> None:
    # Load dataset
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
        eval_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.workers
    )

    # Train autoencoder
    auto_encoder = AutoEncoder((28, 28))
    trainer_autoencoder = pl.Trainer(
        gpus=1 if cfg.device == "cuda" else 0,
        max_epochs=cfg.unsupervised_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            VisualizeEmbeddings(num_batches=15),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer_autoencoder.fit(auto_encoder, train_loader_full, eval_loader)

    # Train classifier
    train_dataset_reduced = torch.utils.data.Subset(
        train_dataset_full, torch.arange(0, cfg.num_train_labels)
    )
    train_loader_reduced = torch.utils.data.DataLoader(
        train_dataset_reduced,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
    )
    classifier = Classifier(auto_encoder)
    trainer_classifier = pl.Trainer(
        gpus=1 if cfg.device == "cuda" else 0,
        max_epochs=cfg.classifier_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer_classifier.fit(classifier, train_loader_reduced, eval_loader)
    result = trainer_classifier.validate(classifier, eval_loader)[0]
    print(result)
    return result
