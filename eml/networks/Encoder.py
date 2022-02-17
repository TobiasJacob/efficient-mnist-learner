from typing import List, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        channels: List[int] = [6, 10, 20],
    ) -> None:
        super().__init__()
        # Encoder
        encoder = []
        for i in range(len(channels)):
            encoder.append(nn.Conv2d(1 if i == 0 else channels[i - 1], channels[i], 3))
            encoder.append(nn.ReLU())
            encoder.append(nn.BatchNorm2d(channels[i]))
            encoder.append(nn.MaxPool2d((2, 2), return_indices=True))
        self.encoder = nn.ModuleList(encoder)

        # Middle part (should be 20)
        x = torch.zeros((1, 1, *image_size))
        x = self(x, simulate=True)
        self.fc_size = x.flatten().shape[0]
        fc_layers = [
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_size),
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_size),
            nn.Linear(self.fc_size, self.fc_size),
        ]
        self.fc_layers = nn.ModuleList(fc_layers)

    def forward(
        self, x: torch.Tensor, simulate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pool_indices = []
        layer_sizes = []
        for layer in self.encoder:
            if type(layer) is nn.MaxPool2d:
                layer_sizes.append(x.shape)
                x, ind = layer(x)
                pool_indices.append(ind)
            else:
                x = layer(x)
        orig_shape_2d = x.shape
        if simulate:
            return x
        x = x.reshape(x.shape[0], self.fc_size)
        for layer in self.fc_layers:
            x = layer(x)
        return x, pool_indices, layer_sizes, orig_shape_2d
