from typing import Type

import torch
import torch.nn as nn


class DownsampleUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float,
        non_linearity: Type,
    ) -> None:
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            non_linearity(inplace=True),
        )
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (3, 3),
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            non_linearity(inplace=True),
            nn.Dropout(dropout, inplace=False),
            nn.Conv2d(
                out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
        )
        self.downsample = nn.Conv2d(
            in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Downsample", x.shape, self.downsample(x).shape)
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)
