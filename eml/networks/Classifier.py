from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score

from eml.networks.AutoEncoder import AutoEncoder


class Classifier(pl.LightningModule):
    def __init__(
        self,
        auto_encoder: AutoEncoder,
        lr: float,
        output_classes: int = 10,
    ) -> None:
        super().__init__()
        self.auto_encoder = auto_encoder
        fc_size = auto_encoder.encoder.fc_size

        # Classifier
        classifier = [
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size),
            nn.Linear(fc_size, output_classes),
        ]
        self.classifier = nn.ModuleList(classifier)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.auto_encoder(x)
        for layer in self.classifier:
            x = layer(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
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

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
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

    def configure_optimizers(self) -> Tuple:
        return self.optimizer
