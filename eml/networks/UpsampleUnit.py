import torch
import torch.nn as nn


class UpsampleUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, dropout: float
    ) -> None:
        super(UpsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (3, 3),
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=False),
            nn.Conv2d(
                out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
        )
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Upsample", x.shape, self.upsample(x).shape)
        x = self.norm_act(x)
        return self.block(x) + self.upsample(x)
