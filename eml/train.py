import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader

from eml.callbacks.VisualizeEmbeddings import VisualizeEmbeddings
from eml.config import Config, config_description
from eml.networks.AutoEncoder import AutoEncoder
from eml.networks.Classifier import Classifier


def train(
    cfg: Config,
    train_loader_full: DataLoader,
    train_loader_reduced: DataLoader,
    eval_loader: DataLoader,
) -> None:
    # Logging
    log_name = config_description(cfg, Config())
    print(log_name)
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name=log_name)

    # Train autoencoder
    auto_encoder = AutoEncoder((28, 28), lr=cfg.lr, channels=cfg.channels)
    trainer_autoencoder = pl.Trainer(
        gpus=1 if cfg.device == "cuda" else 0,
        max_epochs=cfg.unsupervised_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            VisualizeEmbeddings(num_batches=15),
            LearningRateMonitor("epoch"),
        ],
        logger=logger,
        log_every_n_steps=1,
    )
    trainer_autoencoder.fit(auto_encoder, train_loader_full, eval_loader)

    # Train classifier
    classifier = Classifier(auto_encoder, lr=cfg.lr)
    trainer_classifier = pl.Trainer(
        gpus=1 if cfg.device == "cuda" else 0,
        max_epochs=cfg.classifier_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
        ],
        logger=logger,
        log_every_n_steps=1,
    )
    trainer_classifier.fit(classifier, train_loader_reduced, eval_loader)
    result = trainer_classifier.validate(classifier, eval_loader)[0]
    return result
