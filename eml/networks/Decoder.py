from typing import List

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        channels: List[int] = [6, 10, 20],
    ) -> None:
        super().__init__()
        decoder = []
        for i in reversed(range(len(channels))):
            out_features = 1 if i == 0 else channels[i - 1]
            decoder.append(nn.MaxUnpool2d((2, 2)))
            decoder.append(nn.ConvTranspose2d(channels[i], out_features, 3))
            if i != 0:
                decoder.append(nn.ReLU())
                decoder.append(nn.BatchNorm2d(out_features))
        self.decoder = nn.ModuleList(decoder)

    def forward(
        self,
        x: torch.Tensor,
        pool_indices: List[torch.Tensor],
        layer_sizes: List[torch.Tensor],
        orig_shape_2d: torch.Tensor,
    ) -> torch.Tensor:
        x = x.reshape(orig_shape_2d)
        for layer in self.decoder:
            if type(layer) is nn.MaxUnpool2d:
                x = layer(x, pool_indices.pop(), layer_sizes.pop())
            else:
                x = layer(x)
        return x