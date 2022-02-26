import os
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader

from eml.callbacks.VisualizeEmbeddings import VisualizeEmbeddings
from eml.Config import Config, config_description
from eml.networks.AutoEncoder import AutoEncoder
from eml.networks.Classifier import Classifier


def train(
    cfg: Config,
    train_loader_full: DataLoader,
    train_loader_reduced: DataLoader,
    eval_loader: DataLoader,
) -> Dict:
    """Trains a model with the desired configuration and stores the result.

    Args:
        cfg (Config): The global configuration.
        train_loader_full (DataLoader): The full trainig data loader for the
            autoencoder.
        train_loader_reduced (DataLoader): The reduced training data loader for the
            classifier.
        eval_loader (DataLoader): The evaluation dataloader.

    Returns:
        Dict: A dictionary containing 'classifier/val_acc' and 'classifier/val_f1'
    """
    # Logging
    log_name = config_description(cfg, None)
    print(log_name)
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=log_name)

    # Train autoencoder
    auto_encoder = AutoEncoder(
        cfg=cfg,
    )
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
    classifier = Classifier(auto_encoder, cfg=cfg)
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

    # Validate
    result = trainer_classifier.validate(classifier, eval_loader)[0]
    with open(f"{log_name}-result.txt", "w") as f:
        f.write(str(result))
    return result
