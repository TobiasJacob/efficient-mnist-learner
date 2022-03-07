from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy, f1_score

from eml.Config import Config, get_non_linearity
from eml.networks.AutoEncoder import AutoEncoder
from eml.networks.FCUnit import FCUnit
from eml.sam.sam import SAM
from eml.sam.smooth_cross_entropy import smooth_crossentropy
from eml.sam.step_lr import StepLR


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
        self.cfg = cfg

        # Classifier
        fc_size = cfg.autoencoder_features
        classifier = []
        for i in range(cfg.classifier_size):
            classifier.append(FCUnit(fc_size, cfg.dropout_p, get_non_linearity(cfg)))
        classifier.append(nn.Linear(fc_size, output_classes))
        self.classifier = nn.Sequential(*classifier)

        # Optimizer
        if cfg.use_sam:
            self.optimizer = SAM(
                self.parameters(),
                torch.optim.SGD,
                rho=cfg.sam_rho,
                adaptive=cfg.sam_adaptive,
                lr=cfg.classifier_lr,
                momentum=cfg.sam_momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=cfg.classifier_lr, weight_decay=cfg.weight_decay
            )
        self.automatic_optimization = False

        if cfg.advanced_initialization:
            self._initialize()

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
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
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
        x = self.classifier(x)
        preds = torch.argmax(x, dim=-1)
        loss = smooth_crossentropy(x, y).mean()
        self.log("classifier/train_loss", loss)
        self.log("classifier/train_acc", accuracy(preds, y))
        self.log("classifier/train_f1", f1_score(preds, y, num_classes=10))
        self.log("classifier/lr_clas", self.optimizer.param_groups[0]["lr"])

        if self.cfg.use_sam:
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            x, y = batch
            x = self.auto_encoder(x)
            x = self.classifier(x)
            preds = torch.argmax(x, dim=-1)
            loss = smooth_crossentropy(x, y).mean()
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def on_epoch_end(self) -> None:
        self.lr_scheduler.step()
        return super().on_epoch_end()

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
        x = self.classifier(x)
        preds = torch.argmax(x, dim=-1)
        loss = smooth_crossentropy(x, y)
        self.log("classifier/val_loss", loss)
        self.log("classifier/val_acc", accuracy(preds, y))
        self.log("classifier/val_f1", f1_score(preds, y, num_classes=10))
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """The optimizer for this network.

        Returns:
            torch.optim.Optimizer: The optimizer for this network.
        """
        self.lr_scheduler = StepLR(
            self.optimizer,
            self.cfg.classifier_lr,
            self.cfg.classifier_epochs,
        )
        return [
            {
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
            },
        ]

    def _initialize(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight.data, mode="fan_in", nonlinearity="relu"
                )
                m.bias.data.zero_()
        nn.init.kaiming_normal_(
            self.classifier[-1].weight.data,
            mode="fan_in",
            nonlinearity="linear",
        )
