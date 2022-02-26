from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter

from eml.Config import Config
from eml.networks.Decoder import Decoder
from eml.networks.Encoder import Encoder


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
        self.variational_sigma = cfg.variational_sigma
        self.encoder = Encoder(
            image_size, cfg.auto_encoder_fc_layers, cfg.auto_encoder_channels
        )
        self.decoder = Decoder(
            self.encoder.fc_size,
            cfg.auto_encoder_fc_layers,
            cfg.auto_encoder_channels,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.autoencoder_lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the images into a feature vector.

        Args:
            x (torch.Tensor): The images from the dataset.
                Shape: (batch_size, width, height)

        Returns:
            torch.Tensor: The encoded features.
                Shape: (batch_size, self.encoder.fc_size)
        """
        (x, _, _, _) = self.encoder(x)
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
        x, y = batch
        z, pool_indices, layer_sizes, orig_shape_2d = self.encoder(x)
        # if this is a variational autoencoder, add noise
        if self.variational_sigma is not None:
            z += torch.randn_like(z) * self.variational_sigma
        x_hat = self.decoder(z, pool_indices, layer_sizes, orig_shape_2d)
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
        loss, x, x_hat = self.full_forward(batch)
        self.log("autoencoder/train_loss", loss)
        if batch_idx % 200 == 0:
            tensorboard: SummaryWriter = self.logger.experiment
            grid = self.visualize_reconstructions(x, x_hat)
            tensorboard.add_image(
                "train/reconstructions",
                grid,
                self.global_step,
            )

        return loss

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
        return self.optimizer
