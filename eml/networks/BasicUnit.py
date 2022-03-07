from typing import Type

import torch
import torch.nn as nn


class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float, non_linearity: Type) -> None:
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            non_linearity(inplace=True),
            nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
