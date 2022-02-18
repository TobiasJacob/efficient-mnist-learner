from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter

from eml.networks.Decoder import Decoder
from eml.networks.Encoder import Encoder


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        image_size: Tuple[int, int],
        channels: List[int] = [6, 10, 20],
    ) -> None:
        super().__init__()
        self.encoder = Encoder(image_size, channels)
        self.decoder = Decoder(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (x, _, _, _) = self.encoder(x)
        return x

    def visualize_reconstructions(
        self, input_imgs: torch.Tensor, reconst_imgs: torch.Tensor
    ) -> None:
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
        grid = torchvision.utils.make_grid(
            imgs[:16], nrow=4, normalize=True, range=(-1, 1)
        )
        return grid

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(*z)
        loss = F.mse_loss(x_hat, x)
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
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(*z)
        loss = F.mse_loss(x_hat, x)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
