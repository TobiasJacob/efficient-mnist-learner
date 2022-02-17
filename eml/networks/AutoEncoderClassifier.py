from typing import List, Tuple

import torch
import torch.nn as nn


class AutoEncoderClassifier(nn.Module):
    def __init__(
        self, image_size: Tuple[int, int], channels: List[int] = [6, 10, 20]
    ) -> None:
        super().__init__()
        self.image_size = image_size
        down_convolutions = []
        for i in range(len(channels)):
            down_convolutions.append(
                nn.Conv2d(1 if i == 0 else channels[i - 1], channels[i], 3)
            )
        self.down_convolutions = nn.ModuleList(down_convolutions)

        up_convolutions = []
        for i in reversed(range(len(channels))):
            up_convolutions.append(
                nn.ConvTranspose2d(channels[i], 1 if i == 0 else channels[i - 1], 3)
            )
        self.up_convolutions = nn.ModuleList(up_convolutions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.down_convolutions:
            x = layer(x)
        for layer in self.up_convolutions:
            x = layer(x)
        return x
