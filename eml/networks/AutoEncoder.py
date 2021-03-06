from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter

from eml.Config import Config, get_non_linearity
from eml.networks.Decoder import Decoder
from eml.networks.Encoder import Encoder
from eml.sam.sam import SAM
from eml.sam.step_lr import StepLR


class AutoEncoder(pl.LightningModule):
    """The variational AutoEncoder module."""

    def __init__(
        self,
        cfg: Config,
        image_size: Tuple[int, int] = (28, 28),
    ) -> None:
        """Creates a new AutoEncoder.

        Args:
            cfg (Config): Configuration options for the AutoEncoder.
            image_size (Tuple[int, int], optional): Image size. Defaults to (28, 28).
        """
        super().__init__()
        self.cfg = cfg
        self.variational_sigma = cfg.variational_sigma
        self.encoder = Encoder(
            image_size,
            cfg.auto_encoder_fc_layers,
            cfg.auto_encoder_channels,
            cfg.dropout_p,
            cfg.auto_encoder_depth,
            cfg.autoencoder_features,
            get_non_linearity(cfg),
            cfg.autoencoder_stride,
        )
        self.decoder = Decoder(
            cfg.autoencoder_features,
            self.encoder.fc_size,
            cfg.auto_encoder_fc_layers,
            cfg.auto_encoder_channels,
            cfg.dropout_p,
            cfg.auto_encoder_depth,
            get_non_linearity(cfg),
            cfg.autoencoder_stride,
        )

        if cfg.use_sam:
            self.optimizer = SAM(
                self.parameters(),
                torch.optim.SGD,
                rho=cfg.sam_rho,
                adaptive=cfg.sam_adaptive,
                lr=cfg.sam_autoencoder_lr,
                momentum=cfg.sam_momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=cfg.autoencoder_lr, weight_decay=cfg.weight_decay
            )
        self.automatic_optimization = False

        if cfg.advanced_initialization:
            self._initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the images into a feature vector.

        Args:
            x (torch.Tensor): The images from the dataset.
                Shape: (batch_size, width, height)

        Returns:
            torch.Tensor: The encoded features.
                Shape: (batch_size, self.encoder.fc_size)
        """
        (x, _) = self.encoder(x)
        return x

    def full_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Applies a full forward pass, which encodes and reconstructs the images.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The images and labels from the
                dataset. The labels are not used.

        Returns:
            torch.Tensor: The reconstructed images after being processed by the
            autoencoder. Shape: (batch_size, width, height)
        """
        if self.cfg.autoencoder_stride != 3:
            return None
        x, y = batch
        z, orig_shape_2d = self.encoder(x)
        # if this is a variational autoencoder, add noise
        if self.variational_sigma is not None:
            z += torch.randn_like(z) * self.variational_sigma
        x_hat = self.decoder(z, orig_shape_2d)
        loss = F.mse_loss(x_hat, x)
        return loss, x, x_hat

    def visualize_reconstructions(
        self, input_imgs: torch.Tensor, reconst_imgs: torch.Tensor
    ) -> torch.Tensor:
        """Helper method to render reconstructed iamges into a grid for tensorboard.

        Args:
            input_imgs (torch.Tensor): The original images.
                Shape: (batch_size, width, height)
            reconst_imgs (torch.Tensor): The reconstructed images.
                Shape: (batch_size, width, height)

        Returns:
            torch.Tensor: The images rendered into a grid.
                Shape: (batch_size, width, height)
        """
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
        grid = torchvision.utils.make_grid(
            imgs[:16], nrow=4, normalize=True, range=(-1, 1)
        )
        return grid

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Runs the full forward pass and calculates the loss for a training sample.
        The full forward pass includes encoding and reconstructing the images.
        If batch_idx is a multiple of 200, saves the reconstructed images.

        Args:
            batch (torch.Tensor): The training batch containing images.
                Shape: (batch_size, width, height)
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss for this training sample. Shape: (1)
        """
        if self.cfg.autoencoder_stride != 3:
            return None
        loss, x, x_hat = self.full_forward(batch)
        if self.cfg.use_sam:
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            loss, x, x_hat = self.full_forward(batch)
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.log("autoencoder/train_loss", loss)
        self.log("autoencoder/lr", self.optimizer.param_groups[0]["lr"])
        if batch_idx % 200 == 0:
            tensorboard: SummaryWriter = self.logger.experiment
            grid = self.visualize_reconstructions(x, x_hat)
            tensorboard.add_image(
                "train/reconstructions",
                grid,
                self.global_step,
            )

    def on_epoch_end(self) -> None:
        self.lr_scheduler.step()
        return super().on_epoch_end()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Runs the full forward pass and calculates the loss for a validation sample.
        The full forward pass includes encoding and reconstructing the images.
        If batch_idx is a multiple of 100, saves the reconstructed images.

        Args:
            batch (torch.Tensor): The validation batch containing images.
                Shape: (batch_size, width, height)
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss for this validation sample.
        """
        if self.cfg.autoencoder_stride != 3:
            return None
        loss, x, x_hat = self.full_forward(batch)
        self.log("autoencoder/val_loss", loss)
        if batch_idx % 100 == 0:
            tensorboard: SummaryWriter = self.logger.experiment
            grid = self.visualize_reconstructions(x, x_hat)
            tensorboard.add_image(
                "val/reconstructions",
                grid,
                self.global_step,
            )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """The optimizer for this network.

        Returns:
            torch.optim.Optimizer: The optimizer for this network.
        """
        self.lr_scheduler = StepLR(
            self.optimizer, self.cfg.autoencoder_lr, self.cfg.unsupervised_epochs
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }

        self._initialize()

    def _initialize(self) -> None:
        for m in self.modules():
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
            self.encoder.fc_layers[-1].weight.data, mode="fan_in", nonlinearity="linear"
        )
        nn.init.kaiming_normal_(
            self.decoder.fc_layers[0].weight.data, mode="fan_in", nonlinearity="linear"
        )
