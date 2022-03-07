from typing import Type

import torch
import torch.nn as nn


class FCUnit(nn.Module):
    def __init__(self, features: int, dropout: float, non_linearity: Type) -> None:
        super(FCUnit, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(features),
            non_linearity(inplace=True),
            nn.Linear(features, features),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
