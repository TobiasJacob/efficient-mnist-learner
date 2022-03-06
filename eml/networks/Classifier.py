from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score

from eml.Config import Config
from eml.networks.AutoEncoder import AutoEncoder
from eml.networks.FCUnit import FCUnit


class Classifier(pl.LightningModule):
    """Classifier which takes in the encoded image feature vectors and classifies them
    into different types of clothing. The classifier uses a provided AutoEncoder module
    to generate the feature embeddings from the images.
    """

    def __init__(
        self,
        auto_encoder: AutoEncoder,
        cfg: Config,
        output_classes: int = 10,
    ) -> None:
        """Creates a new classifier.

        Args:
            auto_encoder (AutoEncoder): The AutoEncoder which should be used to
            calculate the embeddings.
            cfg (Config): The global config object.
            output_classes (int, optional): Number of output classes. Defaults to 10.
        """
        super().__init__()
        self.auto_encoder = auto_encoder

        # Classifier
        fc_size = cfg.autoencoder_features
        classifier = []
        for i in range(cfg.classifier_size):
            classifier.append(FCUnit(fc_size, cfg.dropout_p))

        classifier.append(nn.Linear(fc_size, output_classes))
        self.classifier = nn.Sequential(*classifier)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.classifier_lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the forward pass, which inclused encoding and classifying the images.

        Args:
            x (torch.Tensor): The images. Shape: (batch_size, width, height).

        Returns:
            torch.Tensor: The class probabilities. Shape: (batch_size, output_classes).
        """
        x = self.auto_encoder(x)
        for layer in self.classifier:
            x = layer(x)
        return x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Runs the forward pass and calculates the loss for a training sample.
        The full forward pass includes encoding and classifing the images.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The training batch
                images and labels. Shape: (batch_size, width, height) and (batch_size)
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss for this training sample. Shape: (1)
        """
        x, y = batch
        x = self.auto_encoder(x)
        for layer in self.classifier:
            x = layer(x)
        preds = torch.argmax(x, dim=-1)
        loss = F.cross_entropy(x, y)
        self.log("classifier/train_loss", loss)
        self.log("classifier/train_acc", accuracy(preds, y))
        self.log("classifier/train_f1", f1_score(preds, y, num_classes=10))
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Runs the forward pass and calculates the loss for a validation sample.
        The full forward pass includes encoding and classifing the images.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The validation batch images
                and labels. Shape: (batch_size, width, height) and (batch_size)
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss for this validation sample. Shape: (1)
        """
        x, y = batch
        x = self.auto_encoder(x)
        for layer in self.classifier:
            x = layer(x)
        preds = torch.argmax(x, dim=-1)
        loss = F.cross_entropy(x, y)
        self.log("classifier/val_loss", loss)
        self.log("classifier/val_acc", accuracy(preds, y))
        self.log("classifier/val_f1", f1_score(preds, y, num_classes=10))
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """The optimizer for this network.

        Returns:
            torch.optim.Optimizer: The optimizer for this network.
        """
        return self.optimizer
