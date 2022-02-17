from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

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

    def training_step(self, batch: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(*z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(*z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
