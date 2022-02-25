from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score

from eml.config import Config
from eml.networks.AutoEncoder import AutoEncoder


class Classifier(pl.LightningModule):
    def __init__(
        self,
        auto_encoder: AutoEncoder,
        cfg: Config,
        output_classes: int = 10,
    ) -> None:
        super().__init__()
        self.auto_encoder = auto_encoder

        # Classifier
        classifier = []
        for i in range(len(cfg.classifier_neurons)):
            if i == 0:
                layer_in_neurons = auto_encoder.encoder.fc_size
            else:
                layer_in_neurons = cfg.classifier_neurons[i - 1]
            classifier.append(nn.Linear(layer_in_neurons, cfg.classifier_neurons[i]))
            classifier.append(nn.ReLU())
            classifier.append(nn.BatchNorm1d(cfg.classifier_neurons[i]))

        classifier.append(nn.Linear(cfg.classifier_neurons[-1], output_classes))
        self.classifier = nn.ModuleList(classifier)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.classifier_lr)

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
