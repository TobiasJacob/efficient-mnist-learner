from typing import List, Tuple

import torch
import torch.nn as nn


class AutoEncoderClassifier(nn.Module):
    def __init__(
        self, image_size: Tuple[int, int], channels: List[int] = [6, 10, 20]
    ) -> None:
        super().__init__()

        # Size type
        encoder = []
        for i in range(len(channels)):
            encoder.append(nn.Conv2d(1 if i == 0 else channels[i - 1], channels[i], 3))
            encoder.append(nn.ReLU())
            encoder.append(nn.BatchNorm2d(channels[i]))
            encoder.append(nn.MaxPool2d((2, 2), return_indices=True))
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        for i in reversed(range(len(channels))):
            out_features = 1 if i == 0 else channels[i - 1]
            decoder.append(nn.MaxUnpool2d((2, 2)))
            decoder.append(nn.ConvTranspose2d(channels[i], out_features, 3))
            if i != 0:
                decoder.append(nn.ReLU())
                decoder.append(nn.BatchNorm2d(out_features))
        self.decoder = nn.ModuleList(decoder)

        # Find middle size (should be 20)
        x = torch.zeros((1, 1, *image_size))
        (x, _, _) = self.encode(x)
        self.fc_size = x.flatten().shape[0]
        self.fc_layers = [
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_size),
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_size),
            nn.Linear(self.fc_size, self.fc_size),
        ]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pool_indices = []
        layer_sizes = []
        for layer in self.encoder:
            if type(layer) is nn.MaxPool2d:
                layer_sizes.append(x.shape)
                x, ind = layer(x)
                pool_indices.append(ind)
            else:
                x = layer(x)
        return x, pool_indices, layer_sizes

    def decode(
        self,
        x: torch.Tensor,
        pool_indices: List[torch.Tensor],
        layer_sizes: List[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.decoder:
            if type(layer) is nn.MaxUnpool2d:
                x = layer(x, pool_indices.pop(), layer_sizes.pop())
            else:
                x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (x, pool_indices, layer_sizes) = self.encode(x)
        orig_shape = x.shape
        x = x.reshape(-1, self.fc_size)
        for layer in self.fc_layers:
            x = layer(x)
        x = x.reshape(orig_shape)
        x = self.decode(x, pool_indices, layer_sizes)
        return x
