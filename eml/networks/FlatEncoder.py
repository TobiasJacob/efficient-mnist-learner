from typing import Tuple

import pytorch_lightning as pl
import torch


class AutoEncoder(pl.LightningModule):
    def __init__(self, image_size: Tuple[int, int], out_size: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(image_size[0] * image_size[1], out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x
